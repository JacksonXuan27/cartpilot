from dataclasses import dataclass
from enum import Enum


class TopicLabel(str, Enum):
    RETURN_EXCHANGE = "退换货"
    LOGISTICS = "物流"
    SIZE = "尺码"
    INVOICE = "发票"
    QUALITY_ISSUE = "质量问题"
    SHIPPING_FEE = "运费"
    PROMOTION = "优惠活动"
    PRICE_PROTECTION = "价保"
    PAYMENT = "支付"
    ORDER_MODIFICATION = "订单修改"
    RESTOCK = "库存补货"
    PRODUCT_INFORMATION = "商品信息"
    WARRANTY_REPAIR = "保修维修"
    ACCOUNT = "账号"
    MEMBERSHIP_POINTS = "会员积分"
    REVIEW = "评价"
    OTHER = "其他"


@dataclass(frozen=True, slots=True)
class TopicClass:
    label: TopicLabel
    boundary: str
    examples: tuple[str, ...]
    severity: str


TOPIC_CLASSES: tuple[TopicClass, ...] = (
    TopicClass(TopicLabel.RETURN_EXCHANGE, "退货、换货、退款怎么办；维修归保修维修", ("退货", "退款", "退钱", "七天无理由还能退不"), "strict"),
    TopicClass(TopicLabel.LOGISTICS, "货走到哪了、什么时候送到；运费费用归运费", ("快递", "发货", "到哪了", "海外直邮"), "strict"),
    TopicClass(TopicLabel.SIZE, "商品大小、码数是否合适", ("猫窝买大了", "项圈偏码", "适合几斤的猫"), "strict"),
    TopicClass(TopicLabel.INVOICE, "开票、抬头和报销凭证", ("开发票", "发票抬头开错了", "合并开票"), "strict"),
    TopicClass(TopicLabel.QUALITY_ISSUE, "商品本身的破损、瑕疵或故障", ("开胶", "破了个洞", "有瑕疵"), "strict"),
    TopicClass(TopicLabel.SHIPPING_FEE, "运费承担和运费险理赔，不包含物流轨迹", ("包邮吗", "退货运费谁承担", "运费险怎么赔"), "medium"),
    TopicClass(TopicLabel.PROMOTION, "优惠券和促销活动的使用规则", ("优惠券", "满减", "能叠加用吗"), "medium"),
    TopicClass(TopicLabel.PRICE_PROTECTION, "购买后降价的补差价诉求", ("刚买就降价了", "能补差价吗", "保价期多久"), "medium"),
    TopicClass(TopicLabel.PAYMENT, "付款和支付过程中的问题", ("付不了款", "扣了两次钱", "货到付款"), "medium"),
    TopicClass(TopicLabel.ORDER_MODIFICATION, "下单后修改信息或取消订单", ("改地址", "改电话号码", "订单还能取消吗"), "medium"),
    TopicClass(TopicLabel.RESTOCK, "商品库存和补货时间", ("有货吗", "断货了", "什么时候补货"), "medium"),
    TopicClass(TopicLabel.PRODUCT_INFORMATION, "商品材质、功能、用法和选择咨询", ("什么材质", "怎么洗", "猫粮怎么选"), "medium"),
    TopicClass(TopicLabel.WARRANTY_REPAIR, "保修期限、维修和换新；退款退货归退换货", ("保修多久", "坏了能修吗", "能换新吗"), "medium"),
    TopicClass(TopicLabel.ACCOUNT, "登录、绑定和账号安全", ("登录不上", "忘了密码", "注销账号"), "broad"),
    TopicClass(TopicLabel.MEMBERSHIP_POINTS, "会员权益和积分规则", ("积分怎么用", "会员几级", "积分能抵钱吗"), "broad"),
    TopicClass(TopicLabel.REVIEW, "商品评价、追评和晒单规则", ("评价怎么改", "追评在哪写", "晒单有奖励吗"), "broad"),
    TopicClass(TopicLabel.OTHER, "无法归入其他主题的请求", ("闲聊", "转人工", "客服几点上班"), "broad"),
)

TOPIC_LABELS: tuple[TopicLabel, ...] = tuple(item.label for item in TOPIC_CLASSES)
LABEL2ID: dict[TopicLabel, int] = {label: index for index, label in enumerate(TOPIC_LABELS)}
ID2LABEL: dict[int, TopicLabel] = {index: label for index, label in enumerate(TOPIC_LABELS)}
NUM_TOPIC_LABELS = len(TOPIC_LABELS)
