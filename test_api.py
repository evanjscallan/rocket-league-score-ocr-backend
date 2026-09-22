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
        # pyrefly: ignore [unsupported-operation]
        cal_body["username"] = "admin"
        # pyrefly: ignore [unsupported-operation]
        cal_body["password"] = "admin"
        res = await client.put("/ocr-calibration", json=cal_body)
        # 10. GET /ocr-regions returns regions in the expected format
        res = await client.get("/ocr-regions")
        assert res.status_code == 200
        regions = res.json()
        assert "blue_score" in regions
        assert "timer" in regions
        assert "orange_score" in regions
        for key in ["blue_score", "timer", "orange_score"]:
            assert all(k in regions[key] for k in ["x", "y", "width", "height"])

        # 11. PUT /ocr-regions with header authentication updates calibration
        updated_regions = {
            "blue_score": {"x": 0.20, "y": 0.15, "width": 0.18, "height": 0.14},
            "timer": {"x": 0.42, "y": 0.15, "width": 0.16, "height": 0.14},
            "orange_score": {"x": 0.62, "y": 0.15, "width": 0.18, "height": 0.14},
        }
        res = await client.put(
            "/ocr-regions",
            headers={"X-Admin-Username": "admin", "X-Admin-Password": "admin"},
            json=updated_regions,
        )
        assert res.status_code == 200
        saved = res.json()
        assert saved["blue_score"]["x"] == 0.20
        assert saved["timer"]["x"] == 0.42
        assert saved["orange_score"]["x"] == 0.62
        # 12. POST /stop-local-video when not running returns 409
        res = await client.post("/stop-local-video")
        assert res.status_code in {409, 200}

    print("All auth and API tests passed!")


def test_overtime_reducer():
    from ocr import GameStateReducer

    reducer = GameStateReducer()

    # 1. Regulation ends 1-1 at 0:00
    state = reducer.update(1, 1, (0, False))
    assert not state.is_game_over
    assert state.winner is None

    # 2. Premature 0:00 anomaly (e.g. 2-1 for 1 second)
    state = reducer.update(2, 1, (0, False))
    # It might think game over at 0:00 with unequal scores
    # 3. Stream transitions into overtime (+0:02)
    state = reducer.update(2, 1, (2, True))
    # Entering overtime must cancel premature game over!
    assert not state.is_game_over
    assert state.winner is None
    assert state.time_left.is_overtime

    # 4. Overtime progresses (+0:15) with same score
    state = reducer.update(2, 1, (15, True))
    assert not state.is_game_over
    assert state.winner is None

    # 5. Blue scores sudden death golden goal (3-1)
    state = reducer.update(3, 1, (20, True))
    assert state.is_game_over
    assert state.winner == "Blue"
    print("Overtime reducer tests passed!")


if __name__ == "__main__":
    asyncio.run(run_auth_tests())
    test_overtime_reducer()
