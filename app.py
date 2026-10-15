"""سَكينة: مساعد الدعم النفسي باللغة العربية (RAG + GPT). واجهة Streamlit عربية دون أيقونات.

التشغيل:  streamlit run app.py          (تقييم النظام للمطوّر في صفحة منفصلة: streamlit run evaluation.py)

"""
from __future__ import annotations

import html

import streamlit as st

from rag import config
from rag.generation import ALWAYS_NOTE
from rag.pipeline import assistant_index, respond
from rag.safety import CRISIS_MESSAGE, SUPPORT_LINES

st.set_page_config(page_title="سكينة", layout="wide", initial_sidebar_state="collapsed")

from ui import inject_css  # noqa: E402

inject_css()
esc = html.escape


@st.cache_resource(show_spinner="نجهّز قاعدة المعرفة لأول مرة")
def index():
    return assistant_index()


retriever, info = index()

st.markdown(f"""
<div class="top">
  <div>
    <div class="name">سَكينة</div>
    <div class="tagline">مساحة هادئة تكتب فيها ما تشعر به، وتجد ردوداً مبنية على إجابات مختصين نفسيين.</div>
  </div>
  <div class="state">
    <div>{info["cases"]} حالة أجاب عنها مختصون</div>
  </div>
</div>""", unsafe_allow_html=True)

tab_chat, tab_about = st.tabs(["المحادثة", "عن سكينة"])

# ---- conversation ---------------------------------------------------------------------------------
EXAMPLES = [
    "عندي قلق وأرق من شهرين وما أقدر أنام، وش أسوي؟",
    "أحس بضيق وحزن بدون سبب وفقدت الرغبة في كل شيء",
    "تجيني نوبات خوف وخفقان مفاجئة",
    "ابني ما يركز في المدرسة وكثير الحركة",
]


def crisis_html() -> str:
    lines = "".join(f'<div class="line"><b>{n}</b><span>{esc(d)}</span></div>' for n, d in SUPPORT_LINES)
    return f'<div class="crisis"><div class="say">{esc(CRISIS_MESSAGE)}</div><div class="lines">{lines}</div></div>'


def reply_html(r) -> str:
    rep = r.reply
    if rep.mode == "llm":
        head = f'<div class="say">{esc(rep.answer)}</div>'
    else:
        head = (f'<div class="label">أقرب رد كتبه مختص على حالة تشبه حالتك</div>'
                f'<div class="say">{esc(rep.answer)}</div>')
    steps = ("<ol>" + "".join(f"<li>{esc(s)}</li>" for s in rep.steps) + "</ol>") if rep.steps else ""
    tags = "".join(f'<span class="tag">{esc(t)}</span>' for t in r.type_names) or '<span class="tag">عام</span>'
    cases = "".join(f'<div class="case"><div class="q">{esc(h.chunk.case.question[:230])}</div>'
                    f'<div class="a">{esc(h.chunk.case.answer)}</div></div>' for h in rep.hits)
    return (f'<div class="reply">{head}{steps}'
            f'<div class="meta">موضوع رسالتك {tags}</div>'
            f'<div class="record"><div class="title">حالات حقيقية مشابهة وردود المختصين عليها</div>{cases}</div>'
            f'<div class="fine">{esc(ALWAYS_NOTE)}</div>'
            + (f'<div class="gpt-off">لم يكتب GPT شرحاً لهذا الرد: {esc(rep.error)}</div>' if rep.error else "")
            + '</div>')


st.session_state.setdefault("chat", [])

with tab_chat:
    if not st.session_state["chat"]:
        st.markdown('<div class="welcome"><h2>بماذا تشعر اليوم؟</h2>'
                    '<p>اكتب بكلماتك، بالفصحى أو العامية. سكينة لا تشخّص ولا تصف أدوية، لكنها تريك ما قاله مختصون '
                    'لأشخاص مرّوا بشيء مشابه، وتساعدك على الخطوة التالية.</p></div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex{i}", width="stretch"):
                st.session_state["pending"] = ex
                st.rerun()
    else:
        for r in st.session_state["chat"]:
            st.markdown(f'<div class="you">{esc(r.message)}</div>', unsafe_allow_html=True)
            st.markdown(crisis_html() if r.crisis else reply_html(r), unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True, border=False):
        msg = st.text_area("رسالتك", placeholder="اكتب ما تشعر به هنا", height=110, label_visibility="collapsed")
        _, c2 = st.columns([5, 1])
        sent = c2.form_submit_button("أرسل", width="stretch")
    text = st.session_state.pop("pending", None) or (msg.strip() if sent and msg.strip() else None)
    if text:
        with st.spinner("نقرأ رسالتك ونبحث في حالات مشابهة"):
            st.session_state["chat"].append(respond(text))
        st.rerun()
    if st.session_state["chat"] and st.button("ابدأ محادثة جديدة"):
        st.session_state["chat"] = []
        st.rerun()
    st.markdown('<div class="help-strip">إذا كنت في خطر الآن اتصل بالإسعاف <b>997</b>، '
                'أو بمركز الاستشارات النفسية <b>920033360</b>، أو بوزارة الصحة <b>937</b> على مدار الساعة.</div>',
                unsafe_allow_html=True)

# ---- about ----------------------------------------------------------------------------------------
with tab_about:
    st.markdown(f"""
<div class="sec" style="margin-top:0"><h3>المشكلة</h3></div>
<p class="prose">كثيرون يعانون من القلق أو الاكتئاب أو الأرق ولا يعرفون من أين يبدؤون. الوصمة والتكلفة والانتظار تدفعهم للبحث في الإنترنت،
فيجدون منتديات غير موثوقة، أو روبوتات محادثة عامة قد تخترع معلومات طبية ولا تذكر مصدرها ولا تنتبه حين يكون الشخص في خطر.</p>
<div class="sec"><h3>الحل</h3></div>
<p class="prose">تردّ سكينة فقط من ردود حقيقية كتبها مختصون نفسيون على حالات مشابهة، وتعرض هذه الحالات ليتحقق الشخص بنفسه، وتتعرف على نوع
رسالته، وتشجع دائماً على مراجعة مختص. وقبل كل رد تفحص الرسالة: إن ظهرت فيها أفكار انتحار أو إيذاء للنفس تعرض رسالة دعم وأرقام المساعدة.</p>
<div class="sec"><h3>ما تكتبه وما تحصل عليه</h3></div>
<table class="deflist">
<tr><td>تكتب</td><td>ما تشعر به أو سؤالك، بالفصحى أو العامية.</td></tr>
<tr><td>تحصل على</td><td>رد داعم مختصر يصوغه GPT من ردود المختصين، وخطوات عملية، وموضوع رسالتك، وثلاث حالات حقيقية مشابهة مع ردود المختصين عليها.</td></tr>
<tr><td>في حالة الأزمة</td><td>رسالة دعم وأرقام 937 و920033360 و997 و1919 بدل الرد العادي.</td></tr>
</table>
<div class="sec"><h3>البيانات</h3></div>
<p class="prose" style="font-size:1rem">MentalQA (Alhuzali وآخرون، IEEE Access 2024، رخصة MIT، ضمن مسابقة AraHealthQA 2025): 500 سؤال حقيقي
بالعربية أجاب عنها مختصون في الصحة النفسية. <a href="https://github.com/hasanhuz/MentalQA" target="_blank">github.com/hasanhuz/MentalQA</a></p>
<div class="sec"><h3>كيف تعمل</h3></div>
<table class="deflist">
<tr><td>التضمين</td><td>{esc(config.EMBEDDING_MODEL)} يحوّل كل حالة إلى متجه يمثل معناها. المستخدم الآن: {esc(info["embedder"])}.</td></tr>
<tr><td>قاعدة المتجهات</td><td>ChromaDB تحفظ المتجهات وتبحث بالتشابه.</td></tr>
<tr><td>البحث</td><td>هجين: المعنى (المتجهات) مع الكلمات (BM25 و TF-IDF)، ثم أقرب {config.TOP_K} حالات.</td></tr>
<tr><td>الصياغة</td><td>{esc(config.LLM_MODEL)} من OpenAI يكتب الرد من ردود المختصين فقط: دون تشخيص، ودون أدوية أو جرعات، مع الإحالة لمختص.</td></tr>
</table>
""", unsafe_allow_html=True)

st.markdown('<div class="footer">سكينة أداة دعم وتوعية، وليست بديلاً عن الطبيب أو الأخصائي النفسي، ولا تقدم تشخيصاً ولا علاجاً.</div>',
            unsafe_allow_html=True)
