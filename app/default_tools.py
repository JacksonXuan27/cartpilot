from datetime import datetime, timezone

from app.after_sales_tools import (
    AfterSalesTool,
    InMemoryAfterSalesRepository,
    InMemoryRefundRepository,
    RefundTool,
)
from app.faq_tools import FAQEntry, FAQSearchTool, InMemoryFAQRepository
from app.logistics_tools import (
    InMemoryShipmentRepository,
    LogisticsQueryTool,
    Shipment,
    TrackingEvent,
)
from app.order_tools import InMemoryOrderRepository, Order, OrderQueryTool
from app.tool_registry import ToolRegistry


_DEMO_ORDER_ID = "ORD-DEMO-1001"
_DEMO_TIME = datetime(2026, 9, 25, 10, 30, tzinfo=timezone.utc)


def default_tool_registry() -> ToolRegistry:
    """Build the deterministic business tools used by the local app runtime."""
    order_repository = InMemoryOrderRepository(
        {
            _DEMO_ORDER_ID: Order(
                order_id=_DEMO_ORDER_ID,
                status="shipped",
                product_name="CartPilot 演示耳机",
                quantity=1,
                total_amount=399.0,
                currency="CNY",
                placed_at=_DEMO_TIME,
            )
        }
    )
    shipment_repository = InMemoryShipmentRepository(
        {
            _DEMO_ORDER_ID: Shipment(
                order_id=_DEMO_ORDER_ID,
                carrier="CartPilot Express",
                tracking_number="CPDEMO1001",
                status="in_transit",
                estimated_delivery=datetime(
                    2026, 9, 28, 18, 0, tzinfo=timezone.utc
                ),
                events=[
                    TrackingEvent(
                        occurred_at=_DEMO_TIME,
                        location="上海",
                        description="包裹已交给承运商，正在运输中",
                    )
                ],
            )
        }
    )
    faq_repository = InMemoryFAQRepository(
        [
            FAQEntry(
                faq_id="FAQ-DEMO-1001",
                category="refund",
                title="退款到账时间",
                answer="退款申请审核通过后，通常在三个工作日内原路退回。",
                keywords=("退款", "到账", "三个工作日"),
            ),
            FAQEntry(
                faq_id="FAQ-DEMO-1002",
                category="logistics",
                title="物流异常处理",
                answer="物流超过预计时间未更新时，可以先查询运单并联系人工客服核实。",
                keywords=("物流", "运单", "异常"),
            ),
        ]
    )
    return ToolRegistry(
        [
            OrderQueryTool(order_repository),
            LogisticsQueryTool(shipment_repository),
            FAQSearchTool(faq_repository),
            RefundTool(order_repository, InMemoryRefundRepository()),
            AfterSalesTool(order_repository, InMemoryAfterSalesRepository()),
        ]
    )
