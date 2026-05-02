from turba_models import list_models


def test_list_models_has_expected_fields():
    models = list_models()
    assert len(models) == 5
    for item in models:
        assert "crop_name" in item
        assert "model_name" in item
        assert item["feature_columns"] == [
            "longitude",
            "latitude",
            "soil_ph",
            "organic_matter_pct",
            "available_p2o5",
            "available_k2o",
        ]
