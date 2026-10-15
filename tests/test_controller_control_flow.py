import time
import threading
import unittest
from unittest.mock import patch, Mock

from explore_skill.controller import ExploreController, TaskHandle


class _Grid:
    @staticmethod
    def world_to_cell(_x, _y):
        return (4, 7)


class ExploreControlFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = ExploreController(
            map_topic="/map",
            nav_navigate_endpoint="http://127.0.0.1:1/mcp/",
            nav_status_endpoint="http://127.0.0.1:1/mcp/",
            nav_cancel_endpoint="http://127.0.0.1:1/mcp/",
        )
        self.controller._latest_pose_xyyaw = (1.0, 2.0, 0.0)
        self.controller._latest_map = object()
        self.reading = (0, 1, 1)
        self.controller._reference = Mock(read=lambda: self.reading)
        self.controller._map_reference = self.controller._pose_reference = self.reading

    def _handle(self, legs: int = 0) -> TaskHandle:
        handle = TaskHandle(
            task_id="test", started_at=0, timeout_s=60.0,
            max_speed_m_s=0.1, legs_completed=legs,
            reference=self.reading,
        )
        self.controller._task = handle
        return handle

    @patch("explore_skill.frontier.GridView.from_msg", return_value=_Grid())
    def test_sweep_requires_periodic_leg_and_missing_coverage(self, _grid) -> None:
        self.controller._viewed_sectors[(4, 7)] = set()
        self.assertFalse(self.controller._should_sweep(self._handle(1)))
        self.assertFalse(self.controller._should_sweep(self._handle(2)))
        self.assertTrue(self.controller._should_sweep(self._handle(3)))

        self.controller._viewed_sectors[(4, 7)] = set(range(6))
        self.assertFalse(self.controller._should_sweep(self._handle(3)))

    def test_navigation_timeout_cancels_the_accepted_run(self) -> None:
        calls = []

        def rpc(tool, args):
            calls.append((tool, args))
            if tool == "status" and any(t == "cancel" for t, _ in calls):
                return {"known": True, "state": "CANCELED"}
            if tool == "navigate":
                return {"accepted": True, "run_id": "run-42"}
            if tool == "status":
                self.reading = (self.reading[0] + 1_000_000, 1, 1)
                return {"state": "RUNNING"}
            return {"accepted": True}

        self.controller._mcp_call_sync = rpc
        self.controller.NAV_POLL_PERIOD_S = 0.001
        ok, detail = self.controller._nav_navigate_blocking(
            1.0, 2.0, yaw=None, timeout_s=0.003,
            cancel_evt=self._handle(),
        )

        self.assertFalse(ok)
        self.assertEqual(detail, "leg timeout")
        self.assertIn(("cancel", {"run_id": "run-42"}), calls)

    def test_navigation_cancel_request_cancels_the_accepted_run(self) -> None:
        calls = []

        def rpc(tool, args):
            calls.append((tool, args))
            if tool == "status" and any(t == "cancel" for t, _ in calls):
                return {"known": True, "state": "CANCELED"}
            if tool == "navigate":
                handle.cancel_requested = True
                return {"accepted": True, "run_id": "run-7"}
            return {"accepted": True} if tool == "cancel" else {"state": "RUNNING"}

        handle = self._handle()
        self.controller._mcp_call_sync = rpc
        ok, detail = self.controller._nav_navigate_blocking(
            1.0, 2.0, yaw=None, timeout_s=1.0, cancel_evt=handle,
        )

        self.assertFalse(ok)
        self.assertEqual(detail, "canceled during nav")
        self.assertIn(("cancel", {"run_id": "run-7"}), calls)

    def test_lost_submission_response_cancels_only_its_owned_run(self):
        import uuid
        handle = self._handle()
        calls = []
        self.controller._mcp_call_sync = lambda tool, args: calls.append((tool, args)) or ({"known": True, "state": "CANCELED"} if tool == "status" else {})
        with self.assertRaisesRegex(RuntimeError, "before acknowledgment"):
            self.controller._nav_navigate_blocking(0, 0, yaw=None, timeout_s=1, cancel_evt=handle)
        uuid.UUID(handle.nav_run_id)
        self.assertIn(("cancel", {"run_id": handle.nav_run_id}), calls)

    def test_failed_cancellation_reports_error_instead_of_canceled(self):
        handle = self._handle()
        handle.nav_run_id = "owned"
        handle.cancel_requested = True
        self.controller._mcp_call_sync = lambda *_: {}
        self.controller._run_frontiers = Mock(side_effect=RuntimeError("canceled"))
        with patch("explore_skill.controller.time.monotonic", side_effect=[0, 6]):
            self.controller._run_task(handle)
        self.assertEqual(handle.state, "error")
        self.assertIn("cancellation unacknowledged", handle.detail)

    def test_cancel_from_mcp_loop_only_requests_worker_cancellation(self):
        import asyncio
        handle = self._handle()
        handle.nav_run_id = "owned"
        self.controller._nav_cancel_rpc = Mock()
        async def request():
            accepted, _ = self.controller.cancel(handle.task_id)
            self.assertTrue(accepted)
        asyncio.run(request())
        self.assertTrue(handle.cancel_requested)
        self.controller._nav_cancel_rpc.assert_not_called()

    def test_partial_runtime_cleanup_closes_reference_despite_node_error(self):
        self.controller._node = Mock(destroy_node=Mock(side_effect=RuntimeError("node cleanup")))
        with self.assertRaisesRegex(RuntimeError, "node cleanup"):
            self.controller.stop_runtime()
        self.controller._reference.close.assert_called_once()

    def test_map_wait_uses_reference_zero_and_allows_delayed_delivery(self):
        self._handle()
        self.controller._latest_map = None
        result = []
        thread = threading.Thread(target=lambda: result.append(self.controller._wait_for_map(.01)))
        thread.start()
        try:
            time.sleep(.05)
            self.assertTrue(thread.is_alive())
            frame = object()
            self.controller._on_map(frame)
            thread.join(.5)
            self.assertEqual(result, [frame])
        finally:
            self.controller._stop_evt.set()
            thread.join(.5)

    def test_paused_leg_keeps_budget_and_reset_cancels_its_run(self):
        handle = self._handle()
        calls = []

        def rpc(tool, args):
            calls.append((tool, args))
            if tool == "status" and any(t == "cancel" for t, _ in calls):
                return {"known": True, "state": "CANCELED"}
            return {"accepted": True, "run_id": "owned"} if tool == "navigate" else ({"accepted": True} if tool == "cancel" else {"state": "RUNNING"})

        self.controller._mcp_call_sync = rpc
        self.controller.NAV_POLL_PERIOD_S = .005
        self.controller._run_frontiers = lambda h: self.controller._nav_navigate_blocking(
            0, 0, yaw=None, timeout_s=.01, cancel_evt=h)
        thread = threading.Thread(target=self.controller._run_task, args=(handle,))
        thread.start()
        try:
            time.sleep(.05)
            self.assertTrue(thread.is_alive())
            self.assertEqual(self.controller.status(None)["elapsed_s"], 0)
            self.reading = (0, 2, 1)
            thread.join(.5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(handle.state, "error")
            self.assertIn(("cancel", {"run_id": "owned"}), calls)
        finally:
            self.controller._stop_evt.set()
            thread.join(.5)

    def test_quiet_done_requires_new_map_evidence(self):
        handle = self._handle()
        self.controller._sweep_360 = Mock()
        with patch("explore_skill.frontier.GridView.from_msg"), \
                patch("explore_skill.frontier.mapped_free_area_m2", return_value=0), \
                patch("explore_skill.frontier.total_frontier_count", return_value=0), \
                patch("explore_skill.frontier.pick_target", return_value=None):
            thread = threading.Thread(target=self.controller._run_task, args=(handle,))
            thread.start()
            try:
                time.sleep(.05)
                self.assertEqual(handle.state, "exploring")
                self.assertEqual(self.controller._sweep_360.call_count, 1)
                self.reading = (31_000_000_000, 1, 1)
                time.sleep(.15)
                self.assertEqual(handle.state, "exploring")
                self.controller._on_map(object())
                thread.join(.5)
                self.assertEqual(handle.state, "done")
            finally:
                self.controller._stop_evt.set()
                thread.join(.5)


if __name__ == "__main__":
    unittest.main()
