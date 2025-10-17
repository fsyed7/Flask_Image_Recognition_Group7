# test_acceptance_happy.py

from io import BytesIO
import pytest
from PIL import Image

def test_acceptance_successful_upload(client):
    """
    Test Case: Successful Upload of a Valid Image File
    - Purpose: Ensure the application accepts a valid image file upload and provides a prediction.
    - Method:
        - Create a mock valid image file with minimal valid data.
        - Simulate a POST request to the `/prediction` route with the file.
        - Assert the response status code is 200.
        - Verify that the response data includes the keyword 'Prediction.'
    """
    img_data = BytesIO(b"fake_image_data")  # Simulated valid image data
    img_data.name = "test_image.jpg"

    response = client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"Prediction" in response.data


def test_acceptance_valid_large_image(client):
    """
    Test Case: Upload of a Valid Large Image File
    - Purpose: Check if the system accepts large but valid image files without errors and still provides predictions.
    - Method:
        - Create a mock large image file by repeating mock image data multiple times.
        - Simulate a POST request to the `/prediction` route with the file.
        - Assert the response status code is 200.
        - Verify the presence of 'Prediction' in the response data.
    """
    img_data = BytesIO(b"fake_large_image_data" * 1000)  # Simulating a large image
    img_data.name = "large_image.jpg"

    response = client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"Prediction" in response.data


def test_acceptance_valid_image_size_upload(client):
    """
    Test Case: Upload of an Image with a Specific Large Size
    - Purpose: Validate system behavior with valid image files of a specific size or resolution.
    - Method:
        - Simulate an image upload with mock data representing a large image.
        - POST the file to the `/prediction` route.
        - Check that the status code is 200 and 'Prediction' exists in the response.
    """
    img_data = BytesIO(b"valid_image_data_of_large_size" * 1000)  # Simulating a specific size
    img_data.name = "large_image.jpg"

    response = client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"Prediction" in response.data


# -------------------------
# New tests (2 basic / 2 advanced)
# -------------------------


def test_basic_missing_file_field_returns_error(client):
    """
    Basic test:
    - POST without 'file' field should be handled and return the error page.
    - Expect status 200 and the error message displayed.
    """
    response = client.post(
        "/prediction",
        data={},  # no file field
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data
    assert b"Prediction" in response.data  # page still renders the Prediction header


def test_basic_non_image_file_returns_error(client):
    """
    Basic test:
    - Upload a non-image file (text) and expect the app to handle it gracefully.
    - The response should show the same error message.
    """
    txt = BytesIO(b"This is not an image")
    txt.name = "not_image.txt"

    response = client.post(
        "/prediction",
        data={"file": (txt, txt.name)},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_advanced_valid_image_monkeypatched_prediction(client, monkeypatch):
    """
    Advanced test:
    - Use a real in-memory JPEG image so PIL can open it.
    - Monkeypatch the route's prediction function to return a deterministic value
      (avoids dependency on model file or heavy prediction).
    - Verify the returned page contains the mocked prediction value.
    """
    # Create a simple in-memory image
    img = Image.new("RGB", (224, 224), color="white")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    buf.name = "valid.jpg"

    # Monkeypatch the predict_result used by the app to a deterministic integer
    monkeypatch.setattr("app.predict_result", lambda _img: 3)

    response = client.post(
        "/prediction",
        data={"file": (buf, buf.name)},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"Prediction" in response.data
    assert b"3" in response.data  # mocked prediction displayed


def test_advanced_consistent_prediction_across_requests(client, monkeypatch):
    """
    Advanced test:
    - Monkeypatch prediction to return a numpy integer to ensure the app handles numpy types.
    - Send multiple identical requests and verify consistent results.
    """
    import numpy as np

    # Create a simple in-memory image
    img = Image.new("RGB", (224, 224), color="white")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    buf.name = "valid2.jpg"

    # Return a numpy integer
    monkeypatch.setattr("app.predict_result", lambda _img: np.int32(5))

    # First request
    buf_first = BytesIO(buf.getvalue())
    buf_first.name = buf.name
    buf_first.seek(0)

    resp1 = client.post(
        "/prediction",
        data={"file": (buf_first, buf_first.name)},
        content_type="multipart/form-data"
    )

    # Second request (fresh stream)
    buf_second = BytesIO(buf.getvalue())
    buf_second.name = buf.name
    buf_second.seek(0)

    resp2 = client.post(
        "/prediction",
        data={"file": (buf_second, buf_second.name)},
        content_type="multipart/form-data"
    )

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert b"5" in resp1.data
    assert b"5" in resp2.data
    assert resp1.data == resp2.data  # pages should be identical for same input / mock
