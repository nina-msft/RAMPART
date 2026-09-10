# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the canonical trace/result serializer."""

from __future__ import annotations

import json
import math
import re
from dataclasses import fields
from datetime import (
    UTC,
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter

from rampart.core.result import (
    InjectionRecord,
    PopulationRef,
    Result,
    SafetyStatus,
)
from rampart.core.serialization import (
    TRACE_SCHEMA_VERSION,
    ResultRecord,
    SchemaError,
    UnsupportedSchemaVersionError,
    deserialize_result,
    serialize_result,
)
from rampart.core.types import (
    EvalOutcome,
    EvalResult,
    ObservabilityLevel,
    Payload,
    PayloadFormat,
    Request,
    Response,
    SideEffect,
    ToolCall,
    Turn,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

_TIMESTAMP = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _make_eval_result() -> EvalResult:
    return EvalResult(
        outcome=EvalOutcome.DETECTED,
        confidence=0.75,
        evidence=["saw the thing", "and another"],
        rationale="because reasons",
        undetermined_operands=["left operand undetermined"],
    )


def _make_turn() -> Turn:
    request = Request(
        prompt="do the thing",
        attachments=[
            Payload(
                content="poisoned doc text",
                id="payload-1",
                format=PayloadFormat.MARKDOWN,
                metadata={"persona": "attacker"},
            ),
        ],
    )
    response = Response(
        text="agent said this",
        tool_calls=[
            ToolCall(
                name="send_email",
                arguments={"to": "a@b.com", "nested": {"count": 2}},
                result="ok",
                timestamp=_TIMESTAMP,
            ),
        ],
        side_effects=[SideEffect(kind="http_request", details={"url": "http://x"})],
        metadata={"latency_ms": 12},
    )
    return Turn(
        request=request,
        response=response,
        eval_result=_make_eval_result(),
        turn_number=3,
        timestamp=_TIMESTAMP,
        driver_reasoning="escalate",
    )


def _make_full_result(*, metadata: dict | None = None) -> Result:
    return Result(
        status=SafetyStatus.UNSAFE,
        summary="a violation was detected",
        observability_level=ObservabilityLevel.TOOL_AND_SIDE_EFFECTS,
        turns=[_make_turn()],
        duration_seconds=1.5,
        harm_category="prompt_injection",
        strategy="xpia",
        injections=[InjectionRecord(payload_id="payload-1", surface_name="SharePoint")],
        population=PopulationRef(id="pop-1", index=0, size=5, threshold=0.8),
        metadata={"note": "user data", "nested": {"k": [1, 2]}}
        if metadata is None
        else metadata,
    )


def _minimal_record_dict() -> dict:
    return {
        "version": TRACE_SCHEMA_VERSION,
        "result": {
            "status": "safe",
            "summary": "clean",
            "observability_level": "response_only",
        },
    }


def _freeform_maps(result: Result) -> list[MutableMapping[str, Any]]:
    return [
        result.metadata,
        result.turns[0].request.attachments[0].metadata,
        result.turns[0].response.metadata,
        result.turns[0].response.tool_calls[0].arguments,
        result.turns[0].response.side_effects[0].details,
    ]


class TestRoundTrip:
    def test_full_result_round_trips_to_equal_value(self) -> None:
        original = _make_full_result()
        encoded = ResultRecord(result=original).to_dict()

        decoded = deserialize_result(data=encoded).result

        assert decoded == original

    def test_version_is_stamped_on_the_record(self) -> None:
        encoded = ResultRecord(result=_make_full_result()).to_dict()

        assert encoded["version"] == TRACE_SCHEMA_VERSION
        assert ResultRecord.VERSION == "rampart.trace.v1"

    def test_serialize_result_builds_attribution_collar(self) -> None:
        encoded = serialize_result(
            result=_make_full_result(),
            pytest_nodeid="tests/test_x.py::test_x",
            result_index=2,
        )

        assert encoded["pytest_nodeid"] == "tests/test_x.py::test_x"
        assert encoded["result_index"] == 2

    def test_serialize_result_omits_attribution_when_unset(self) -> None:
        encoded = serialize_result(result=_make_full_result())

        assert "pytest_nodeid" not in encoded
        assert "result_index" not in encoded

    def test_attribution_collar_round_trips(self) -> None:
        encoded = serialize_result(
            result=_make_full_result(),
            pytest_nodeid="tests/test_x.py::test_x",
            result_index=2,
        )

        decoded = deserialize_result(data=encoded)

        assert decoded.pytest_nodeid == "tests/test_x.py::test_x"
        assert decoded.result_index == 2

    def test_nested_values_survive_the_round_trip(self) -> None:
        decoded = deserialize_result(
            data=ResultRecord(result=_make_full_result()).to_dict()
        ).result

        turn = decoded.turns[0]
        assert turn.request.attachments[0].format is PayloadFormat.MARKDOWN
        assert turn.response.tool_calls[0].arguments == {
            "to": "a@b.com",
            "nested": {"count": 2},
        }
        assert turn.response.tool_calls[0].timestamp == _TIMESTAMP
        assert turn.response.side_effects[0].kind == "http_request"
        assert turn.eval_result is not None
        assert turn.eval_result.outcome is EvalOutcome.DETECTED
        assert decoded.injections[0].surface_name == "SharePoint"
        assert decoded.population == PopulationRef(
            id="pop-1", index=0, size=5, threshold=0.8
        )


class TestFieldExhaustiveness:
    def test_every_field_of_every_type_is_serialized(self) -> None:
        body = ResultRecord(result=_make_full_result()).to_dict()["result"]
        turn = body["turns"][0]

        cases = [
            (Result, body),
            (Turn, turn),
            (Request, turn["request"]),
            (Payload, turn["request"]["attachments"][0]),
            (Response, turn["response"]),
            (ToolCall, turn["response"]["tool_calls"][0]),
            (SideEffect, turn["response"]["side_effects"][0]),
            (EvalResult, turn["eval_result"]),
            (InjectionRecord, body["injections"][0]),
            (PopulationRef, body["population"]),
        ]

        for dataclass_type, encoded in cases:
            expected = {field.name for field in fields(dataclass_type)}
            assert expected == set(encoded), dataclass_type.__name__


class TestVersionDispatch:
    def test_unknown_major_fails_closed(self) -> None:
        data = {"version": "rampart.trace.v2", "result": {}}

        with pytest.raises(UnsupportedSchemaVersionError, match="v2"):
            deserialize_result(data=data)

    def test_missing_version_fails_closed(self) -> None:
        with pytest.raises(UnsupportedSchemaVersionError):
            deserialize_result(data={"result": {}})

    def test_non_mapping_record_fails_closed(self) -> None:
        with pytest.raises(SchemaError, match="mapping"):
            deserialize_result(data=[1, 2, 3])


class TestMigrationTolerance:
    def test_unknown_extra_fields_decode(self) -> None:
        encoded = ResultRecord(result=_make_full_result()).to_dict()
        encoded["future_collar"] = {"anything": True}
        encoded["result"]["future_intrinsic"] = 42

        decoded = deserialize_result(data=encoded).result

        assert decoded.status is SafetyStatus.UNSAFE

    def test_missing_optional_fields_use_defaults(self) -> None:
        decoded = deserialize_result(data=_minimal_record_dict()).result

        assert decoded.status is SafetyStatus.SAFE
        assert decoded.turns == []
        assert decoded.duration_seconds == pytest.approx(0.0)
        assert decoded.harm_category is None
        assert decoded.injections == []
        assert decoded.population is None
        assert decoded.metadata == {}

    def test_malformed_present_list_fails_closed(self) -> None:
        data = _minimal_record_dict()
        data["result"]["turns"] = "not-a-list"

        with pytest.raises(SchemaError, match=r"result\.turns"):
            deserialize_result(data=data)

    def test_incomplete_population_reference_fails_closed(self) -> None:
        data = _minimal_record_dict()
        data["result"]["population"] = {}

        with pytest.raises(SchemaError, match=r"result\.population\.id"):
            deserialize_result(data=data)


class TestValueDomain:
    def test_reserved_metadata_keys_are_stripped(self) -> None:
        result = _make_full_result(
            metadata={"_pytest_nodeid": "x::y", "note": "keep me"},
        )

        encoded = ResultRecord(result=result).to_dict()

        assert encoded["result"]["metadata"] == {"note": "keep me"}

    def test_harm_category_is_passed_through_as_string(self) -> None:
        result = _make_full_result()
        result.harm_category = "custom_product_risk"

        encoded = ResultRecord(result=result).to_dict()
        decoded = deserialize_result(data=encoded).result

        assert encoded["result"]["harm_category"] == "custom_product_risk"
        assert decoded.harm_category == "custom_product_risk"

    def test_non_finite_float_fails_closed(self) -> None:
        result = _make_full_result()
        result.duration_seconds = math.inf

        with pytest.raises(SchemaError, match="duration_seconds"):
            ResultRecord(result=result).to_dict()

    def test_non_json_metadata_fails_closed(self) -> None:
        result = _make_full_result(metadata={"blob": object()})

        with pytest.raises(SchemaError, match="metadata"):
            ResultRecord(result=result).to_dict()

    def test_bad_enum_value_fails_closed_on_decode(self) -> None:
        data = _minimal_record_dict()
        data["result"]["status"] = "not_a_status"

        with pytest.raises(SchemaError, match="status"):
            deserialize_result(data=data)

    def test_non_string_harm_category_fails_closed_on_encode(self) -> None:
        result = _make_full_result()
        result.__dict__["harm_category"] = 42

        with pytest.raises(SchemaError, match="harm_category"):
            ResultRecord(result=result).to_dict()

    def test_non_string_harm_category_fails_closed_on_decode(self) -> None:
        data = _minimal_record_dict()
        data["result"]["harm_category"] = {"category": "custom"}

        with pytest.raises(SchemaError, match="harm_category"):
            deserialize_result(data=data)

    def test_boolean_result_index_fails_before_encoding(self) -> None:
        with pytest.raises(SchemaError, match="result_index"):
            serialize_result(result=_make_full_result(), result_index=True)


class TestBinaryPayloadFailsClosed:
    def test_encoding_a_binary_payload_fails_closed(self, tmp_path) -> None:
        artifact = tmp_path / "doc.pdf"
        artifact.write_bytes(b"%PDF-1.4 fake")
        result = _make_full_result()
        result.turns = [
            Turn(
                request=Request(
                    attachments=[
                        Payload(
                            content="binary doc",
                            format=PayloadFormat.PDF,
                            artifact=artifact,
                        ),
                    ],
                ),
                response=Response(text="ok"),
            ),
        ]

        with pytest.raises(SchemaError, match="binary payload"):
            ResultRecord(result=result).to_dict()

    def test_decoding_a_binary_payload_fails_closed(self) -> None:
        data = _minimal_record_dict()
        data["result"]["turns"] = [
            {
                "request": {
                    "prompt": None,
                    "attachments": [{"content": "x", "id": "p", "format": "pdf"}],
                },
                "response": {"text": "ok"},
            },
        ]

        with pytest.raises(SchemaError, match="binary payload"):
            deserialize_result(data=data)

    @pytest.mark.parametrize("payload_format", ["pdf", "docx", "text"])
    def test_artifact_is_rejected_before_filesystem_access(
        self, payload_format: str
    ) -> None:
        data = serialize_result(result=_make_full_result())
        payload = data["result"]["turns"][0]["request"]["attachments"][0]
        payload.update(format=payload_format, artifact="untrusted-artifact")

        with (
            patch.object(
                Path, "exists", side_effect=AssertionError("filesystem access")
            ),
            pytest.raises(SchemaError, match="artifact"),
        ):
            deserialize_result(data=data)

    def test_live_binary_payload_is_still_supported(self, tmp_path: Path) -> None:
        artifact = tmp_path / "doc.pdf"
        artifact.write_bytes(b"%PDF-1.4 fake")
        payload = Payload(content="doc", format=PayloadFormat.PDF, artifact=artifact)

        assert TypeAdapter(Payload).validate_python(payload) is payload
        assert payload.artifact == artifact


class TestResultAdapter:
    def test_body_methods_round_trip_through_json(self) -> None:
        original = _make_full_result()

        body = original.to_dict()
        restored = Result.from_dict(json.loads(json.dumps(body, allow_nan=False)))

        assert restored == original
        assert isinstance(restored.turns[0], Turn)
        assert "version" not in body
        assert body["turns"][0]["timestamp"] == _TIMESTAMP.isoformat()
        assert body["turns"][0]["response"]["tool_calls"][0]["timestamp"] == (
            _TIMESTAMP.isoformat()
        )

    @pytest.mark.parametrize(
        "timestamp",
        [
            _TIMESTAMP,
            _TIMESTAMP.replace(tzinfo=None),
            _TIMESTAMP.replace(tzinfo=timezone(timedelta(seconds=30))),
            _TIMESTAMP.replace(tzinfo=timezone(timedelta(hours=-5))),
        ],
    )
    def test_python_iso_datetimes_preserve_their_wire_text(
        self, timestamp: datetime
    ) -> None:
        result = _make_full_result()
        result.turns[0].__dict__["timestamp"] = timestamp
        result.turns[0].response.tool_calls[0].timestamp = timestamp

        encoded = serialize_result(result=result)

        assert encoded["result"]["turns"][0]["timestamp"] == timestamp.isoformat()
        assert ResultRecord.from_dict(encoded).result == result
        Draft202012Validator(
            ResultRecord.json_schema(),
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        ).validate(encoded)

    def test_record_filters_only_top_level_metadata_without_mutation(self) -> None:
        original = _make_full_result(
            metadata={
                "_rampart_source_worker": "gw0",
                "_pytest_nodeid": "test",
                "_rampart_worker_artifact_path": object(),
                "user": {"_rampart_source_worker": "keep"},
            }
        )
        record = ResultRecord(result=original)
        original.summary = "updated after wrapping"

        body = record.to_dict()["result"]
        body["metadata"]["user"]["extra"] = True

        assert record.result is original
        assert body["summary"] == original.summary
        assert body["metadata"] == {
            "user": {"_rampart_source_worker": "keep", "extra": True}
        }
        assert original.metadata["user"] == {"_rampart_source_worker": "keep"}
        assert "_rampart_worker_artifact_path" in original.metadata

    def test_body_does_not_own_transport_filtering(self) -> None:
        result = _make_full_result(metadata={"_rampart_source_worker": "gw0"})

        assert result.to_dict()["metadata"] == result.metadata
        assert ResultRecord(result=result).to_dict()["result"]["metadata"] == {}

    @pytest.mark.parametrize("index", [None, 0, 2])
    def test_optional_attribution_is_not_inferred(self, index: int | None) -> None:
        record = ResultRecord(result=_make_full_result(), result_index=index)

        encoded = record.to_dict()

        assert ResultRecord.from_dict(encoded).result_index == index
        assert ("result_index" in encoded) is (index is not None)

    @pytest.mark.parametrize("nodeid", [False, 1, [], {}])
    def test_invalid_nodeid_is_rejected(self, nodeid: object) -> None:
        data = _minimal_record_dict()
        data["pytest_nodeid"] = nodeid

        with pytest.raises(SchemaError, match="pytest_nodeid"):
            ResultRecord.from_dict(data)

    def test_nested_mutations_are_revalidated(self) -> None:
        result = _make_full_result()
        result.turns[0].response.__dict__["text"] = 42

        with pytest.raises(SchemaError, match=r"result\.turns\[0\]\.response\.text"):
            result.to_dict()

    @pytest.mark.parametrize("invalid", [True, 1.5, "1"])
    def test_integer_fields_are_not_coerced(self, invalid: object) -> None:
        result = _make_full_result()
        assert result.population is not None
        result.population.__dict__["index"] = invalid

        with pytest.raises(SchemaError, match=r"result\.population\.index"):
            result.to_dict()

    def test_missing_payload_identity_is_not_generated(self) -> None:
        body = _make_full_result().to_dict()
        del body["turns"][0]["request"]["attachments"][0]["id"]

        with pytest.raises(SchemaError, match="id"):
            Result.from_dict(body)

    def test_invalid_timestamp_is_rejected(self) -> None:
        body = _make_full_result().to_dict()
        body["turns"][0]["timestamp"] = "not a date"

        with pytest.raises(SchemaError, match=r"result\.turns\[0\]\.timestamp"):
            Result.from_dict(body)

    @pytest.mark.parametrize(
        "field", ["turns", "injections", "metadata", "duration_seconds"]
    )
    def test_null_is_not_a_default_for_nonnullable_fields(self, field: str) -> None:
        body = _make_full_result().to_dict()
        body[field] = None

        with pytest.raises(SchemaError, match=field):
            Result.from_dict(body)


class TestJsonValueDomain:
    @pytest.mark.parametrize("map_index", range(5))
    @pytest.mark.parametrize(
        "invalid",
        [
            pytest.param((1, 2), id="tuple"),
            pytest.param(b"bytes", id="bytes"),
            pytest.param(Path("file"), id="path"),
            pytest.param(object(), id="opaque"),
            pytest.param(math.inf, id="infinity"),
            pytest.param(-math.inf, id="negative-infinity"),
            pytest.param(math.nan, id="nan"),
            pytest.param({1: "non-string key"}, id="non-string-key"),
        ],
    )
    def test_freeform_values_are_not_lossily_encoded(
        self, *, map_index: int, invalid: object
    ) -> None:
        result = _make_full_result()
        _freeform_maps(result)[map_index]["nested"] = {"bad": invalid}

        with pytest.raises(SchemaError, match="nested"):
            result.to_dict()

    @pytest.mark.parametrize(
        "invalid",
        [(1, 2), b"bytes", Path("file"), object(), math.inf, math.nan, {1: "x"}],
    )
    def test_dictionary_input_is_checked_before_json_encoding(
        self, invalid: object
    ) -> None:
        body = _make_full_result().to_dict()
        body["metadata"]["bad"] = invalid

        with pytest.raises(SchemaError, match="metadata"):
            Result.from_dict(body)

    def test_cyclic_values_fail_with_a_field_path(self) -> None:
        result = _make_full_result()
        result.metadata["cycle"] = result.metadata

        with pytest.raises(SchemaError, match=r"metadata.*cycle"):
            result.to_dict()
        body = _minimal_record_dict()["result"]
        body["metadata"] = result.metadata
        with pytest.raises(SchemaError, match=r"metadata.*cycle"):
            Result.from_dict(body)

    def test_supported_values_round_trip_without_mutation(self) -> None:
        metadata = {
            "values": [None, True, False, 0, -(2**80), 2**80, 1.25, "text"],
            "nested": {"list": [{"text": "hello"}]},
        }
        result = _make_full_result(metadata=metadata)

        restored = Result.from_dict(result.to_dict())
        restored.metadata["nested"]["list"][0]["text"] = "changed"

        assert result.metadata == metadata
        assert metadata["nested"]["list"][0]["text"] == "hello"
        assert restored.metadata["values"] == metadata["values"]


class TestGeneratedSchema:
    def test_generated_schema_is_valid_and_matches_checked_in_contract(self) -> None:
        schema = ResultRecord.json_schema()
        path = Path(__file__).resolve().parents[3] / "schemas" / "trace.v1.schema.json"

        Draft202012Validator.check_schema(schema)

        assert json.loads(path.read_text(encoding="utf-8")) == schema
        assert schema["properties"]["version"]["const"] == TRACE_SCHEMA_VERSION

    @pytest.mark.parametrize("payload_format", list(PayloadFormat))
    def test_schema_and_decoder_agree_on_payload_formats(
        self, payload_format: PayloadFormat
    ) -> None:
        data = serialize_result(result=_make_full_result())
        data["result"]["turns"][0]["request"]["attachments"][0]["format"] = (
            payload_format.value
        )
        validator = Draft202012Validator(ResultRecord.json_schema())

        assert validator.is_valid(data) is payload_format.is_text
        if payload_format.is_text:
            assert ResultRecord.from_dict(data).result.turns[0].request.attachments
        else:
            with pytest.raises(SchemaError, match="binary payload"):
                ResultRecord.from_dict(data)

    @pytest.mark.parametrize("full", [False, True])
    def test_full_and_minimal_records_conform(self, *, full: bool) -> None:
        data = (
            serialize_result(result=_make_full_result(), result_index=0)
            if full
            else _minimal_record_dict()
        )
        validator = Draft202012Validator(
            ResultRecord.json_schema(),
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        )

        validator.validate(data)
        validator.validate(ResultRecord.from_dict(data).to_dict())

    def test_unknown_additive_fields_are_allowed_at_every_level(self) -> None:
        data = serialize_result(result=_make_full_result())
        body = data["result"]
        turn = body["turns"][0]
        objects = [
            data,
            body,
            turn,
            turn["request"],
            turn["request"]["attachments"][0],
            turn["response"],
            turn["response"]["tool_calls"][0],
            turn["response"]["side_effects"][0],
            turn["eval_result"],
            body["injections"][0],
            body["population"],
        ]
        for item in objects:
            item["future"] = {"recorded": True}

        Draft202012Validator(ResultRecord.json_schema()).validate(data)
        assert ResultRecord.from_dict(data).result == _make_full_result()

    @pytest.mark.parametrize(
        ("path", "invalid"),
        [
            (("result", "summary"), 123),
            (("result", "population", "index"), True),
            (("result", "population", "threshold"), "0.8"),
            (("result", "turns", 0, "request", "prompt"), 123),
            (("result", "turns", 0, "timestamp"), False),
            (("result", "turns", 0, "response", "text"), None),
            (("result", "turns", 0, "response", "tool_calls", 0, "result"), 123),
            (("result", "turns", 0, "request", "attachments", 0, "artifact"), "file"),
            (("result", "turns", 0, "request", "attachments", 0, "format"), "unknown"),
            (("result", "turns", 0, "eval_result", "outcome"), "unknown"),
            (("result", "injections", 0, "payload_id"), 123),
            (("pytest_nodeid",), 123),
            (("result_index",), True),
        ],
    )
    def test_schema_and_decoder_reject_malformed_fields(
        self, *, path: tuple[str | int, ...], invalid: object
    ) -> None:
        data = serialize_result(result=_make_full_result())
        parent: Any = data
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = invalid

        assert not Draft202012Validator(ResultRecord.json_schema()).is_valid(data)
        with pytest.raises(SchemaError, match=re.escape(str(path[-1]))):
            ResultRecord.from_dict(data)

    @pytest.mark.parametrize(
        ("prompt", "attachments", "valid"),
        [
            (None, False, False),
            ("", False, True),
            ("text", False, True),
            (None, True, True),
        ],
    )
    def test_request_invariant_is_in_schema(
        self, *, prompt: str | None, attachments: bool, valid: bool
    ) -> None:
        data = serialize_result(result=_make_full_result())
        request = data["result"]["turns"][0]["request"]
        request["prompt"] = prompt
        if not attachments:
            request["attachments"] = []

        assert Draft202012Validator(ResultRecord.json_schema()).is_valid(data) is valid
        if valid:
            ResultRecord.from_dict(data)
        else:
            with pytest.raises(SchemaError, match="request"):
                ResultRecord.from_dict(data)

    def test_schema_requires_recorded_payload_id(self) -> None:
        data = serialize_result(result=_make_full_result())
        del data["result"]["turns"][0]["request"]["attachments"][0]["id"]

        assert not Draft202012Validator(ResultRecord.json_schema()).is_valid(data)
