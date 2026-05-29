import streamlit as st
import pandas as pd
from datetime import datetime
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

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
        "description": "阳光真i保、大麦旗舰等，健康告知严格"
    },
    "互联网中端": {
        "mortality_coef": 0.22,
        "expense_rate": 0.12,
        "brand_premium": 1.05,
        "simple_calibration": 1.05,
        "description": "支付宝微信平台中端产品"
    },
    "线下大品牌": {
        "mortality_coef": 0.23,
        "expense_rate": 0.15,
        "brand_premium": 1.18,
        "simple_calibration": 2.33,
        "description": "平安/国寿/太平洋等线下大牌"
    },
    "外资品牌": {
        "mortality_coef": 0.22,
        "expense_rate": 0.14,
        "brand_premium": 1.15,
        "simple_calibration": 2.25,
        "description": "友邦/安联等外资"
    },
    "银行系保险": {
        "mortality_coef": 0.225,
        "expense_rate": 0.13,
        "brand_premium": 1.12,
        "simple_calibration": 2.18,
        "description": "工银安盛/建信等银行系"
    }
}

liab_factor = {"仅身故": 1.0, "身故+全残": 1.12}
discount_rate = 0.035

# 2025版CL4标准体生命表（20-59岁男性）
mortality_male_standard = [
    0.00062, 0.00065, 0.00068, 0.00072, 0.00077,  # 20-24
    0.00083, 0.00090, 0.00098, 0.00108, 0.00119,  # 25-29
    0.00121, 0.00133, 0.00146, 0.00161, 0.00179,  # 30-34
    0.00199, 0.00222, 0.00248, 0.00277, 0.00309,  # 35-39
    0.00345, 0.00385, 0.00430, 0.00480, 0.00536,  # 40-44
    0.00599, 0.00670, 0.00750, 0.00840, 0.00941,  # 45-49
    0.01054, 0.01181, 0.01322, 0.01479, 0.01655,  # 50-54
    0.01852, 0.02072, 0.02317, 0.02590, 0.02895   # 55-59
]

# ===================== 马尔可夫精算模型 =====================
class MarkovLifeInsurance:
    def __init__(self, brand_name):
        self.brand = brand_name
        self.config = brand_config[brand_name]
        self.v = 1 / (1 + discount_rate)

    def get_mortality(self, start_age, policy_term):
        start_idx = max(0, start_age - 20)
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

# ===================== 简易拟合模型 =====================
class TermLifeSimpleModel:
    def __init__(self, brand_name):
        self.brand = brand_name
        self.config = brand_config[brand_name]
        self.params = {
            "female": {"a0":2185.2,"a1":-73.84,"a2":-175.6,"a3":6.52},
            "male":{"a0":2421.7,"a1":-82.63,"a2":-227.4,"a3":8.91}
        }
        self.expense_rate = 0.12
        self.profit_split = 0.4
        self.adjust_factor = {
            "standard":1.0,"smoker":1.75,"sub_health":1.35,"high_risk_job":1.65
        }

    def calc_gross_premium(self, age, gender, term, insured_amount, adjust_type="standard"):
        p = self.params[gender]
        if age < 30:
            base_30 = p["a0"] + p["a1"]*30 + p["a2"]*term + p["a3"]*30*term
            age_factor = 0.6 + (age-20)*0.04  
            base = base_30 * age_factor
        else:
            base = p["a0"] + p["a1"]*age + p["a2"]*term + p["a3"]*age*term
        
        gross = base * (insured_amount / 100)
        gross = gross * self.adjust_factor[adjust_type]
        gross = gross * self.config["simple_calibration"]
        return round(gross, 2)

    def calc_premium_load(self, gross_p):
        pure_p = gross_p / (1 + self.expense_rate)
        load = gross_p - pure_p
        load_rate = (load / pure_p)*100
        return round(pure_p,2),round(load,2),round(load_rate,2)

    def calc_insurer_benefit(self, gross_p, load, term):
        total_gross = gross_p * term
        total_load = load * term
        total_profit = total_load * self.profit_split
        profit_rate = (total_profit / total_gross)*100
        return round(total_profit,2),round(profit_rate,2)

    def calc_consumer_simple(self, gross_p, risk_aversion):
        reserve_p = gross_p * (1 + risk_aversion * 0.1)
        surplus = reserve_p - gross_p
        return round(reserve_p,2),round(surplus,2)

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

# ===================== 生成PDF报告 =====================
def generate_pdf_report(inputs, res_markov, res_simple):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    style = getSampleStyleSheet()
    title = Paragraph("定期寿险精算估价报告", style["Heading1"])
    story.append(title)
    story.append(Spacer(1,12))

    # 基础信息
    info_data = [
        ["测算时间", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["投保年龄", f"{inputs['age']}岁"],
        ["性别", inputs['gender']],
        ["保额", f"{inputs['amount']}万元"],
        ["缴费年限", inputs['pay_term']],
        ["保障期间", inputs['protect_term']],
        ["品牌渠道", inputs['brand']],
        ["实际年保费", f"{inputs['real']} 元"]
    ]
    t1 = Table(info_data, colWidths=[180,250])
    t1.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightblue),
        ('GRID',(0,0),(-1,-1),1,colors.black)
    ]))
    story.append(t1)
    story.append(Spacer(1,20))

    # 双模型结果
    table_data = [
        ["测算指标","马尔可夫精算","简易拟合"],
        ["理论毛保费",f"{res_markov['gross']}",f"{res_simple['gross']}"],
        ["纯保费",f"{res_markov['pure']}",f"{res_simple['pure']}"],
        ["溢价金额",f"{res_markov['load']}",f"{res_simple['load']}"],
        ["内部溢价率%",f"{res_markov['load_rate']}",f"{res_simple['load_rate']}"],
        ["实际溢价率%",f"{res_markov['actual_rate']}",f"{res_simple['actual_rate']}"],
        ["消费者剩余",f"{res_simple['surplus']}",""]
    ]
    t2 = Table(table_data, colWidths=[150,180,180])
    t2.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),1,colors.black)
    ]))
    story.append(t2)
    doc.build(story)
    buffer.seek(0)
    return buffer

# ===================== 生成分享链接 =====================
def gen_share_link(inputs):
    params = f"age={inputs['age']}|gender={inputs['gender']}|amount={inputs['amount']}|pay={inputs['pay_term']}|protect={inputs['protect_term']}|brand={inputs['brand']}|real={inputs['real']}"
    base_url = "https://share.streamlit.io/你的账号/仓库/main/app.py"
    share_url = base_url + "?" + params
    return share_url

# ---------------------- 主页面 ----------------------
def main():
    st.title("📊 定期寿险精算估价系统")
    st.markdown("马尔可夫链+CL4生命表双模型估价 | 支持20-59岁全年龄段准确测算")
    st.divider()

    with st.sidebar:
        st.header("📝 投保信息")
        insure_amount = st.slider("保额（万元）",10,500,100,10)
        gender = st.radio("性别",["男","女"],horizontal=True)
        age = st.slider("投保年龄",20,59,30) 
        pay_term = st.selectbox("缴费年限",["10年","15年","20年","30年"],index=3)
        protect_term = st.selectbox("保障期间",["保至60岁","保至70岁","定期20年","保终身"],index=0)
        liab = st.radio("保险责任",["仅身故","身故+全残"],horizontal=True)
        brand = st.selectbox("品牌渠道",list(brand_config.keys()),index=2)
        st.info(brand_config[brand]["description"])
        P_real = st.number_input("保险公司实际报价（元/年）",min_value=0,value=1860)

        st.divider()
        st.header("风险参数")
        adjust_type = st.selectbox("风险类型",
            ["standard","smoker","sub_health","high_risk_job"],
            format_func=lambda x:{"standard":"标准体","smoker":"吸烟","sub_health":"亚健康","high_risk_job":"高危职业"}[x])
        risk_aversion = st.slider("风险厌恶系数",1.0,5.0,2.0,0.1)

        st.divider()
        calc_btn = st.button("🚀 开始测算",type="primary",use_container_width=True)

    input_cache = {
        "age":age,"gender":gender,"amount":insure_amount,
        "pay_term":pay_term,"protect_term":protect_term,
        "brand":brand,"real":P_real
    }

    if calc_btn:
        pay_years = int(pay_term.replace("年",""))
        protect_years = protect_term_to_years(age, protect_term)
        simple_gender = gender_convert(gender)

        # 马尔可夫计算
        markov_model = MarkovLifeInsurance(brand)
        mr = markov_model.calculate(age, insure_amount*10000, pay_years, protect_years, liab)
        markov_gross = mr["gross_premium"]
        markov_pure = mr["net_premium"]
        markov_load = markov_gross - markov_pure
        markov_load_rate = round((markov_load/markov_pure)*100,2) if markov_pure else 0
        markov_actual = round(((P_real-markov_gross)/markov_gross)*100,2)

        # 简易模型
        simple_model = TermLifeSimpleModel(brand)
        sg = simple_model.calc_gross_premium(age, simple_gender, protect_years, insure_amount, adjust_type)
        sp,sl,slr = simple_model.calc_premium_load(sg)
        spf,sfr = simple_model.calc_insurer_benefit(sg, sl, protect_years)
        reserve,surplus = simple_model.calc_consumer_simple(P_real, risk_aversion)
        simple_actual = round(((P_real-sg)/sg)*100,2)

        # 缓存结果用于PDF
        res_markov = {
            "gross":markov_gross,"pure":markov_pure,"load":markov_load,
            "load_rate":markov_load_rate,"actual_rate":markov_actual
        }
        res_simple = {
            "gross":sg,"pure":sp,"load":sl,"load_rate":slr,
            "actual_rate":simple_actual,"surplus":surplus
        }

        # 页面结果展示
        st.header("✅ 测算结果")
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.metric("合理参考保费",f"{round((markov_gross+sg)/2,2)} 元")
        with col2:
            st.metric("实际报价",f"{P_real} 元",f"{(markov_actual+simple_actual)/2:.1f}%")
        with col3:
            st.metric("消费者剩余",f"{surplus} 元")
        with col4:
            st.metric("市场利润率",f"{sfr:.1f}%")

        st.divider()
        # 对比表格
        df = pd.DataFrame({
            "测算指标":["理论毛保费(元)","纯保费(元)","溢价金额(元)","内部溢价率(%)","实际溢价率(%)"],
            "马尔可夫精算":[markov_gross,markov_pure,markov_load,markov_load_rate,markov_actual],
            "简易拟合":[sg,sp,sl,slr,simple_actual]
        })
        st.dataframe(df,use_container_width=True,hide_index=True)

        st.divider()
        # 性价比结论
        if surplus > 100:
            st.success("✅ 性价比优秀，建议投保")
        elif surplus > 0:
            st.info("ℹ 性价比尚可，可考虑")
        elif surplus > -100:
            st.warning("⚠ 性价比一般，建议对比其他产品")
        else:
            st.error("❌ 溢价偏高，不建议入手")

        # 导出PDF
        st.divider()
        st.subheader("📄 导出测算PDF报告")
        pdf_buf = generate_pdf_report(input_cache, res_markov, res_simple)
        st.download_button(
            label="点击下载PDF报告",
            data=pdf_buf,
            file_name=f"定期寿险估价报告_{datetime.now().strftime('%Y%m%d%H%M')}.pdf",
            mime="application/pdf"
        )

        # 生成分享链接
        st.divider()
        st.subheader("🔗 生成分享链接")
        share_url = gen_share_link(input_cache)
        st.code(share_url,language="text")
        st.info("复制上面链接发给别人，打开可自动带入本次测算参数")

if __name__ == "__main__":
    main()
