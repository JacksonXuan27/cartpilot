from dataclasses import dataclass, field
from uuid import uuid4

from app.workflow import WorkflowRuntimeContext, WorkflowState, WorkflowStatus


@dataclass(slots=True)
class RefundInterruptionNode:
    name: str = field(default="refund-confirmation-interrupt", init=False)

    async def execute(
        self,
        state: WorkflowState,
        context: WorkflowRuntimeContext,
    ) -> WorkflowState:
        state.status = WorkflowStatus.RUNNING
        state.current_node = self.name

        if state.data.get("after_sales_requires_confirmation") is not True:
            state.data["confirmation_required"] = False
            state.data["workflow_run_id"] = context.run_id
            return state

        after_sales_info = state.data.get("after_sales_info")
        if not isinstance(after_sales_info, dict):
            after_sales_info = {}

        state.data["confirmation_required"] = True
        state.data["confirmation_id"] = str(uuid4())
        state.data["confirmation_details"] = {
            "scenario": state.data.get("after_sales_scenario"),
            "order_id": after_sales_info.get("order_id"),
            "reason": after_sales_info.get("reason"),
            "requested_action": after_sales_info.get("requested_action"),
        }
        state.data["answer"] = "请确认是否继续此售后申请。"
        state.data["confirmation_status"] = "pending"
        state.status = WorkflowStatus.AWAITING_CONFIRMATION
        state.data["workflow_run_id"] = context.run_id
        return state
