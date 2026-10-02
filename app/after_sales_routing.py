from dataclasses import dataclass, field
from typing import Final

from app.workflow import WorkflowRuntimeContext, WorkflowState, WorkflowStatus


class AfterSalesRoutingError(RuntimeError):
    code = "after_sales_routing_failed"


AFTER_SALES_ROUTES: Final[dict[str, tuple[str, bool]]] = {
    "refund": ("refund_flow", True),
    "return": ("return_flow", True),
    "exchange": ("exchange_flow", True),
    "repair": ("repair_flow", False),
    "logistics_issue": ("logistics_flow", False),
    "other": ("manual_review", True),
    "unknown": ("manual_review", True),
}


@dataclass(slots=True)
class AfterSalesRoutingNode:
    name: str = field(default="after-sales-routing", init=False)

    async def execute(
        self,
        state: WorkflowState,
        context: WorkflowRuntimeContext,
    ) -> WorkflowState:
        state.status = WorkflowStatus.RUNNING
        state.current_node = self.name

        if state.data.get("intent") not in {"refund_return", "after_sales"}:
            state.data["after_sales_routing"] = "skipped"
            state.data["workflow_run_id"] = context.run_id
            return state

        after_sales_intent = state.data.get("after_sales_intent")
        if not isinstance(after_sales_intent, str) or not after_sales_intent.strip():
            error = AfterSalesRoutingError(
                "after-sales intent is required before routing"
            )
            state.status = WorkflowStatus.FAILED
            state.data["error"] = {
                "code": error.code,
                "message": str(error),
            }
            raise error

        route, requires_confirmation = AFTER_SALES_ROUTES.get(
            after_sales_intent,
            ("manual_review", True),
        )
        state.data["after_sales_scenario"] = after_sales_intent
        state.data["after_sales_route"] = route
        state.data["after_sales_requires_confirmation"] = requires_confirmation
        state.data["route"] = route
        if route == "manual_review":
            state.data["after_sales_handoff_reason"] = "unrecognized_after_sales_intent"
        else:
            state.data.pop("after_sales_handoff_reason", None)
        state.data["after_sales_routing"] = "completed"
        state.data["workflow_run_id"] = context.run_id
        return state
