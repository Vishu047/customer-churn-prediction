from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json()['status'] == 'online'

def test_predict_schema_validation():
    invalid_payload = {'gender': 'Alien', 'tenure': -10}
    with TestClient(app) as tc:
        response = tc.post('/predict', json=invalid_payload)
        assert response.status_code == 422
