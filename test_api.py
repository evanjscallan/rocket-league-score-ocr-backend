import asyncio
import base64
from httpx import ASGITransport, AsyncClient
from endpoints import app


async def run_auth_tests():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Unauthenticated request receives 401
        res = await client.get("/ocr-calibration")
        assert res.status_code == 401
        assert res.json()["detail"] == "Admin authentication is required"

        # 2. Query parameter authentication with admin/admin succeeds
        res = await client.get("/ocr-calibration?username=admin&password=admin")
        assert res.status_code == 200

        # 3. Wrong password receives 403
        res = await client.get("/ocr-calibration?username=admin&password=wrongpassword")
        assert res.status_code == 403
        assert res.json()["detail"] == "Invalid admin credentials"

        # 4. Header authentication succeeds
        res = await client.get(
            "/ocr-calibration",
            headers={"X-Admin-Username": "admin", "X-Admin-Password": "admin"},
        )
        assert res.status_code == 200

        # 5. Basic Auth header succeeds
        basic = base64.b64encode(b"admin:admin").decode("ascii")
        res = await client.get(
            "/ocr-calibration",
            headers={"Authorization": f"Basic {basic}"},
        )
        assert res.status_code == 200

        # 6. Admin session creation with username & password
        res = await client.post(
            "/admin/session",
            json={"username": "admin", "password": "admin"},
        )
        assert res.status_code == 200
        assert res.json()["authenticated"] is True
        cookie_header = res.headers.get("set-cookie")
        assert cookie_header is not None

        # 7. Cookie-authenticated session check
        cookie_val = cookie_header.split(";")[0].split("=")[1]
        res = await client.get(
            "/admin/session",
            cookies={"ocr_admin_session": cookie_val},
        )
        assert res.status_code == 200
        assert res.json()["authenticated"] is True

        # 8. PUT endpoint with credentials in query params
        cal_payload = {
            "blue_score": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1},
            "time": {"x": 0.4, "y": 0.1, "width": 0.1, "height": 0.1},
            "orange_score": {"x": 0.8, "y": 0.1, "width": 0.1, "height": 0.1},
        }
        res = await client.put("/ocr-calibration?username=admin&password=admin", json=cal_payload)
        assert res.status_code == 200

        # 9. PUT endpoint with credentials in JSON body
        cal_body = dict(cal_payload)
        cal_body["username"] = "admin"
        cal_body["password"] = "admin"
        res = await client.put("/ocr-calibration", json=cal_body)
        assert res.status_code == 200

    print("All auth tests passed!")


if __name__ == "__main__":
    asyncio.run(run_auth_tests())
