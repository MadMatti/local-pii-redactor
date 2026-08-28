import json

from pii_redactor.evaluation.regex_baseline import regex_detect, regex_response


def test_regex_baseline_returns_exact_source_order_without_overlap() -> None:
    text = (
        "Email ada@example.net, use username amber_user_00001, "
        "and connect from 10.17.0.4."
    )
    detections = regex_detect(text)
    assert [item.type.value for item in detections] == [
        "EMAIL",
        "USERNAME",
        "IP_ADDRESS",
    ]
    assert all(text[item.start : item.end] == item.text for item in detections)
    assert json.loads(regex_response(text)) == {
        "entities": [
            {"type": item.type.value, "text": item.text} for item in detections
        ]
    }


def test_regex_baseline_does_not_treat_empty_field_labels_as_values() -> None:
    text = "The username field and address section are both empty."
    assert regex_detect(text) == ()
