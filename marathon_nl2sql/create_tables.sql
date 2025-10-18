-- 清空并创建交易流水表
DROP TABLE IF EXISTS transaction_flow;
CREATE TABLE transaction_flow (
    transaction_id VARCHAR(64) PRIMARY KEY COMMENT '交易流水唯一标识',
    merchant_id VARCHAR(32) NOT NULL COMMENT '商户ID',
    institution_id VARCHAR(32) NOT NULL COMMENT '发起机构/支付机构ID',
    account_id VARCHAR(32) NOT NULL COMMENT '交易账户ID（付款方）',
    counterparty_id VARCHAR(32) NOT NULL COMMENT '对手方账户ID（收款方）',
    transaction_amount DECIMAL(15,2) NOT NULL COMMENT '交易金额，单位：元，币种为人民币',
    transaction_type VARCHAR(20) NOT NULL COMMENT '交易类型，如 PAYMENT, REFUND',
    transaction_time DATETIME NOT NULL COMMENT '交易发生时间',
    status VARCHAR(15) NOT NULL COMMENT '交易状态，如 SUCCESS, FAILED',
    remark VARCHAR(255) COMMENT '交易备注'
) COMMENT='交易流水表';

-- 清空并创建机构信息表
DROP TABLE IF EXISTS institution_info;
CREATE TABLE institution_info (
    institution_id VARCHAR(32) PRIMARY KEY COMMENT '机构唯一标识',
    institution_name VARCHAR(100) NOT NULL COMMENT '机构名称',
    institution_type VARCHAR(20) NOT NULL COMMENT '机构类型，如 BANK, PAYMENT',
    license_no VARCHAR(50) COMMENT '金融许可证号',
    legal_entity VARCHAR(100) COMMENT '法人代表',
    contact_phone VARCHAR(20) COMMENT '联系电话',
    contact_email VARCHAR(100) COMMENT '联系邮箱',
    status VARCHAR(15) NOT NULL COMMENT '机构状态，如 ACTIVE',
    create_time DATETIME NOT NULL COMMENT '创建时间'
) COMMENT='机构信息表';

-- 清空并创建商户信息表
DROP TABLE IF EXISTS merchant_info;
CREATE TABLE merchant_info (
    merchant_id VARCHAR(32) PRIMARY KEY COMMENT '商户唯一标识',
    merchant_name VARCHAR(100) NOT NULL COMMENT '商户名称',
    merchant_type VARCHAR(20) NOT NULL COMMENT '商户类型，如 RETAIL, ONLINE',
    merchant_category VARCHAR(10) COMMENT '商户类别码 MCC',
    legal_person VARCHAR(50) COMMENT '法人姓名',
    business_license VARCHAR(50) COMMENT '营业执照号',
    settlement_account VARCHAR(32) COMMENT '结算账户ID',
    status VARCHAR(15) NOT NULL COMMENT '商户状态，如 ACTIVE',
    register_time DATETIME NOT NULL COMMENT '注册时间'
) COMMENT='商户信息表';

-- 清空并创建账户信息表
DROP TABLE IF EXISTS account_info;
CREATE TABLE account_info (
    account_id VARCHAR(32) PRIMARY KEY COMMENT '账户唯一标识',
    account_name VARCHAR(100) NOT NULL COMMENT '账户名称',
    account_type VARCHAR(20) NOT NULL COMMENT '账户类型，如 SETTLEMENT',
    bank_name VARCHAR(100) COMMENT '开户银行',
    bank_account_no VARCHAR(34) COMMENT '银行账号',
    status VARCHAR(15) NOT NULL COMMENT '账户状态，如 ACTIVE',
    create_time DATETIME NOT NULL COMMENT '账户开立时间'
) COMMENT='账户信息表';


