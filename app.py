import streamlit as st
import pandas as pd

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="定期寿险精算估价系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- 品牌参数配置中心 ----------------------
brand_config = {
    "互联网高性价比": {
        "mortality_coef": 0.20,
        "expense_rate": 0.08,
        "brand_premium": 1.00,
        "simple_calibration": 1.00,
        "description": "阳光真i保、大麦旗舰版等，价格最低，健康告知严格"
    },
    "互联网中端": {
        "mortality_coef": 0.22,
        "expense_rate": 0.12,
        "brand_premium": 1.05,
        "simple_calibration": 1.05,
        "description": "支付宝、微信等平台产品，服务较好，价格适中"
    },
    "线下大品牌": {
        "mortality_coef": 0.23,
        "expense_rate": 0.15,
        "brand_premium": 1.18,
        "simple_calibration": 2.33,
        "description": "平安、国寿、太平洋等，品牌大，代理人服务，价格较高"
    },
    "外资品牌": {
        "mortality_coef": 0.22,
        "expense_rate": 0.14,
        "brand_premium": 1.15,
        "simple_calibration": 2.25,
        "description": "友邦、安联等，服务优质，品牌溢价高"
    },
    "银行系保险": {
        "mortality_coef": 0.225,
        "expense_rate": 0.13,
        "brand_premium": 1.12,
        "simple_calibration": 2.18,
        "description": "工银安盛、建信人寿等，银行渠道销售，信誉好"
    }
}

# 通用精算参数
liab_factor = {"仅身故": 1.0, "身故+全残": 1.12}
discount_rate = 0.035

# 2025版CL4标准体生命表（30-59岁男性）
mortality_male_standard = [
    0.00121, 0.00133, 0.00146, 0.00161, 0.00179,
    0.00199, 0.00222, 0.00248, 0.00277, 0.00309,
    0.00345, 0.00385, 0.00430, 0.00480, 0.00536,
    0.00599, 0.00670, 0.00750, 0.00840, 0.00941,
    0.01054, 0.01181, 0.01322, 0.01479, 0.01655,
    0.01852, 0.02072, 0.02317, 0.02590, 0.02895
]

# ===================== 模型1：3状态马尔可夫链精算模型 =====================
class MarkovLifeInsurance:
    def __init__(self, brand_name):
        self.brand = brand_name
        self.config = brand_config[brand_name]
        self.v = 1 / (1 + discount_rate)

    def get_mortality(self, start_age, policy_term):
        start_idx = max(0, start_age - 30)
        end_idx = start_idx + policy_term
        base_mort = mortality_male_standard[start_idx:min(end_idx, len(mortality_male_standard))]
        return [m * self.config["mortality_coef"] for m in base_mort]

    def calculate_survival_prob(self, mortality, liability_type):
        n = len(mortality)
        survival_prob = [1.0]
        total_termination_rate = [m * liab_factor[liability_type] for m in mortality]
        
        for t in range(n):
            survival_prob.append(survival_prob[-1] * (1 - total_termination_rate[t]))
        return survival_prob, total_termination_rate

    def calculate_epv_claim(self, survival_prob, total_termination_rate, sum_assured):
        epv_claim = 0
        for t in range(len(total_termination_rate)):
            claim_prob = survival_prob[t] * total_termination_rate[t]
            epv_claim += claim_prob * sum_assured * (self.v ** (t + 1))
        return epv_claim

    def calculate_epv_premium(self, survival_prob, pay_years):
        safe_pay_years = min(pay_years, len(survival_prob))
        epv_premium = 0
        for t in range(safe_pay_years):
            epv_premium += survival_prob[t] * (self.v ** t)
        return epv_premium

    def calculate(self, start_age, sum_assured, pay_years, protect_years, liability_type):
        mortality = self.get_mortality(start_age, protect_years)
        survival_prob, total_termination_rate = self.calculate_survival_prob(mortality, liability_type)
        epv_claim = self.calculate_epv_claim(survival_prob, total_termination_rate, sum_assured)
        epv_premium_denominator = self.calculate_epv_premium(survival_prob, pay_years)
        
        net_premium = epv_claim / epv_premium_denominator
        base_gross = net_premium / (1 - self.config["expense_rate"])
        final_gross = base_gross * self.config["brand_premium"]
        
        return {
            "gross_premium": round(final_gross, 2),
            "net_premium": round(net_premium, 2),
            "epv_claim": round(epv_claim, 2)
        }

# ===================== 模型2：多项式拟合简易模型 =====================
class TermLifeSimpleModel:
    def __init__(self, brand_name):
        self.brand = brand_name
        self.config = brand_config[brand_name]
        
        self.params = {
            "female": {
                "a0": 2185.2,
                "a1": -73.84,
                "a2": -175.6,
                "a3": 6.52
            },
            "male": {
                "a0": 2421.7,
                "a1": -82.63,
                "a2": -227.4,
                "a3": 8.91
            }
        }
        self.expense_rate = 0.12
        self.profit_split = 0.4
        self.adjust_factor = {
            "standard": 1.0,
            "smoker": 1.75,
            "sub_health": 1.35,
            "high_risk_job": 1.65,
            "extra_disease": 1.07,
            "extra_accident": 1.15
        }

    def calc_gross_premium(self, age, gender, term, insured_amount, adjust_type="standard"):
        p = self.params[gender]
        base = p["a0"] + p["a1"] * age + p["a2"] * term + p["a3"] * age * term
        gross = base * (insured_amount / 100)
        gross = gross * self.adjust_factor[adjust_type]
        gross = gross * self.config["simple_calibration"]
        return round(gross, 2)

    def calc_premium_load(self, gross_p):
        pure_p = gross_p / (1 + self.expense_rate)
        load = gross_p - pure_p
        load_rate = (load / pure_p) * 100
        return round(pure_p, 2), round(load, 2), round(load_rate, 2)

    def calc_insurer_benefit(self, gross_p, load, term):
        total_gross = gross_p * term
        total_load = load * term
        total_profit = total_load * self.profit_split
        profit_rate = (total_profit / total_gross) * 100
        return round(total_profit, 2), round(profit_rate, 2)

    def calc_consumer_simple(self, gross_p, risk_aversion):
        reserve_p = gross_p * (1 + risk_aversion * 0.1)
        surplus = reserve_p - gross_p
        return round(reserve_p, 2), round(surplus, 2)

# ---------------------- 工具函数 ----------------------
def gender_convert(gender):
    return "male" if gender == "男" else "female"

def protect_term_to_years(age, protect_term):
    if protect_term == "保至60岁":
        return 60 - age
    elif protect_term == "保至70岁":
        return 70 - age
    elif protect_term == "定期20年":
        return 20
    else:
        return 100 - age

# ---------------------- 主页面 ----------------------
def main():
    # 标题
    st.title("📊 定期寿险精算估价系统")
    st.markdown("基于马尔可夫链精算模型 + 2025版CL4生命表，精准测算定期寿险合理价格与性价比")
    st.divider()

    # 侧边栏输入
    with st.sidebar:
        st.header("📝 产品信息")
        
        insure_amount = st.slider("保额（万元）", 10, 500, 100, 10)
        gender = st.radio("性别", ["男", "女"], horizontal=True)
        age = st.slider("投保年龄", 20, 55, 30)
        
        pay_term = st.selectbox("缴费年限", ["10年", "15年", "20年", "30年"], index=3)
        protect_term = st.selectbox("保障期间", ["保至60岁", "保至70岁", "定期20年", "保终身"], index=0)
        liab = st.radio("保险责任", ["仅身故", "身故+全残"], horizontal=True)
        
        brand = st.selectbox("品牌渠道", list(brand_config.keys()), index=2)
        st.info(brand_config[brand]["description"])
        
        P_real = st.number_input("保险公司实际报价（元/年）", min_value=0, value=1860)
        
        st.divider()
        st.header("👤 个人风险参数")
        adjust_type = st.selectbox("风险类型", 
            ["standard(标准体)", "smoker(吸烟)", "sub_health(亚健康)", "high_risk_job(高危职业)"],
            format_func=lambda x: x.split("(")[0]
        )
        adjust_type = adjust_type.split("(")[0]
        
        risk_aversion = st.slider("风险厌恶系数", 1.0, 5.0, 2.0, 0.1,
            help="1=风险偏好，2=中性，3=轻度厌恶，4=中度厌恶，5=高度厌恶")
        
        calculate_btn = st.button("🚀 开始测算", type="primary", use_container_width=True)

    # 主内容区
    if calculate_btn:
        # 计算参数转换
        pay_years = int(pay_term.replace("年", ""))
        protect_years = protect_term_to_years(age, protect_term)
        simple_gender = gender_convert(gender)

        # 马尔可夫模型计算
        markov_model = MarkovLifeInsurance(brand)
        markov_result = markov_model.calculate(
            start_age=age,
            sum_assured=insure_amount * 10000,
            pay_years=pay_years,
            protect_years=protect_years,
            liability_type=liab
        )
        markov_gross = markov_result["gross_premium"]
        markov_pure = markov_result["net_premium"]
        markov_load = markov_gross - markov_pure
        markov_load_rate = (markov_load / markov_pure) * 100 if markov_pure != 0 else 0
        markov_profit = markov_load * pay_years * 0.4
        markov_profit_rate = (markov_profit / (markov_gross * pay_years)) * 100 if markov_gross != 0 else 0
        markov_actual_rate = ((P_real - markov_gross) / markov_gross) * 100

        # 简易模型计算
        simple_model = TermLifeSimpleModel(brand)
        simple_gross = simple_model.calc_gross_premium(age, simple_gender, protect_years, insure_amount, adjust_type)
        simple_pure, simple_load, simple_load_rate = simple_model.calc_premium_load(simple_gross)
        simple_profit, simple_profit_rate = simple_model.calc_insurer_benefit(simple_gross, simple_load, protect_years)
        simple_reserve, simple_surplus = simple_model.calc_consumer_simple(P_real, risk_aversion)
        simple_actual_rate = ((P_real - simple_gross) / simple_gross) * 100

        # 结果展示
        st.header("✅ 测算结果")
        
        # 核心指标卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("理论合理保费", f"{round((markov_gross+simple_gross)/2, 2)} 元/年", 
                delta=f"马尔可夫: {markov_gross} 元")
        with col2:
            st.metric("实际报价", f"{P_real} 元/年", 
                delta=f"溢价率: {round((markov_actual_rate+simple_actual_rate)/2, 2)}%")
        with col3:
            st.metric("消费者剩余", f"{simple_surplus} 元/年",
                delta="正剩余=性价比好" if simple_surplus>0 else "负剩余=性价比差")
        with col4:
            st.metric("保险公司利润率", f"{round((markov_profit_rate+simple_profit_rate)/2, 2)}%")

        st.divider()

        # 双模型对比表格
        st.subheader("📊 双模型详细对比")
        comparison_data = {
            "指标": ["理论毛保费(元/年)", "纯保费(元/年)", "溢价金额(元/年)", 
                    "内部溢价率(%)", "全周期总利润(元)", "综合利润率(%)", "实际溢价率(%)"],
            "马尔可夫精算模型": [markov_gross, markov_pure, markov_load, 
                              round(markov_load_rate, 2), markov_profit, 
                              round(markov_profit_rate, 2), round(markov_actual_rate, 2)],
            "简易拟合模型": [simple_gross, simple_pure, simple_load, 
                          round(simple_load_rate, 2), simple_profit, 
                          round(simple_profit_rate, 2), round(simple_actual_rate, 2)]
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

        st.divider()

        # 性价比分析
        st.subheader("💡 性价比分析与建议")
        if simple_surplus > 100:
            st.success("✅ 该产品性价比优秀，建议购买")
            st.write("该产品价格低于合理水平，消费者获得了正的剩余价值")
        elif simple_surplus > 0:
            st.info("ℹ️ 该产品性价比良好，可以考虑购买")
            st.write("该产品价格基本合理，符合市场平均水平")
        elif simple_surplus > -100:
            st.warning("⚠️ 该产品性价比一般，建议货比三家")
            st.write("该产品价格略高于合理水平，有一定的品牌溢价")
        else:
            st.error("❌ 该产品性价比偏低，不建议购买")
            st.write("该产品价格明显高于合理水平，存在较高的品牌和渠道溢价")

        st.divider()

        # 品牌参数说明
        with st.expander("📚 精算参数说明"):
            st.write(f"**{brand} 精算参数：**")
            st.write(f"- 死亡率系数：{brand_config[brand]['mortality_coef']}（相对于CL4标准体）")
            st.write(f"- 附加费用率：{brand_config[brand]['expense_rate']*100}%")
            st.write(f"- 品牌溢价系数：{brand_config[brand]['brand_premium']}倍")
            st.write("\n**模型说明：**")
            st.write("- 马尔可夫精算模型：基于3状态马尔可夫链和2025版CL4生命表，计算结果最准确")
            st.write("- 简易拟合模型：基于市场数据的多项式拟合，计算速度快，结果接近实际")

    # 页脚
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666;">
        本系统仅供参考，实际保费以保险公司官方报价为准<br>
        基于开源精算模型开发，免费使用，禁止商业用途
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
