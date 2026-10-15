"""صفحة تقييم النظام للمطوّر (منفصلة عن تطبيق المستخدم).

التشغيل:  streamlit run evaluation.py
"""
from __future__ import annotations

import html
import math

import plotly.express as px
import streamlit as st

from rag import config
from rag.evaluation import load_saved, run_all, save
from ui import CHART, INK, LINE, inject_css

st.set_page_config(page_title="سكينة | تقييم النظام", layout="wide", initial_sidebar_state="collapsed")
inject_css()
esc = html.escape

st.markdown('<div class="top"><div><div class="name">سَكينة</div>'
            '<div class="tagline">صفحة المطوّر: كيف نقيس دقة النظام</div></div></div>'
            '<div style="height:1.4rem;border-bottom:1px solid #DCE2EA;margin-bottom:1.4rem"></div>', unsafe_allow_html=True)

# ---- evaluation -----------------------------------------------------------------------------------
UNIT_AR = {"qa": "السؤال مع رد المختص", "question": "سؤال الشخص فقط", "answer": "رد المختص فقط"}
COL_AR = {"dev_type_p@3": "dev: تطابق النوع (أول 3)", "test_type_p@1": "test: تطابق النوع (الأول)",
          "test_type_p@3": "test: تطابق النوع (أول 3)", "qa_match_hit@1": "الرد الصحيح أولاً",
          "qa_match_hit@5": "الرد الصحيح ضمن 5", "qa_match_mrr": "MRR"}


def nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def stats(entries) -> None:
    """entries: (label, value, note, is_ratio)"""
    out = ""
    for label, v, note, ratio in entries:
        shown = "غير متاح" if nan(v) else (f"{v:.3f}" if ratio else f"{v:,.0f}")
        meter = (f'<div class="meter"><span style="width:{max(0, min(1, v)) * 100:.0f}%"></span></div>'
                 if ratio and not nan(v) else "")
        out += f'<div class="stat"><b>{shown}</b><span>{label}</span>{f"<i>{note}</i>" if note else ""}{meter}</div>'
    st.markdown(f'<div class="stats">{out}</div>', unsafe_allow_html=True)


def section(n: int, title: str, lead: str) -> None:
    st.markdown(f'<div class="sec"><span class="n">{n}</span><h3>{title}</h3></div><p class="lead">{lead}</p>',
                unsafe_allow_html=True)


def chart(df, x, y, color, h=320):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", color_discrete_sequence=CHART)
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Readex Pro", color=INK, size=12),
                      legend=dict(orientation="h", y=1.14, x=1, xanchor="right", title=None), bargap=.3)
    fig.update_xaxes(showgrid=False, title=None, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, title=None, range=[0, 1.02])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})



res = load_saved()
st.markdown('<p class="prose" style="margin:0">كل رقم هنا محسوب على بيانات MentalQA الحقيقية. اختيرت الإعدادات بطريقة '
            '"استبعاد حالة واحدة" على قاعدة المعرفة، وقيست النتائج على ملفات اختبار منفصلة لم يرها النظام. '
            'وبجانب كل نتيجة ما كان سيحققه الاختيار العشوائي، لنعرف هل التحسن حقيقي.</p>', unsafe_allow_html=True)
c1, c2 = st.columns([1, 2])
if c1.button("أعد التقييم الآن", type="primary"):
    with st.spinner("يعمل التقييم، وقد يستغرق دقائق مع نموذج e5 و GPT"):
        save(run_all([config.EMBEDDING_MODEL, "lsa"]))
    st.rerun()
if res:
    c2.markdown(f'<div class="fine" style="margin-top:.7rem">آخر تقييم: {esc(str(res.get("generated_at", "")))}</div>',
                unsafe_allow_html=True)

if not res:
    st.markdown('<div class="notice">لا توجد نتائج بعد. اضغط "أعد التقييم الآن".</div>', unsafe_allow_html=True)
else:
    d, base = res["data"], res["random_baseline"]
    section(1, "البيانات", "أسئلة حقيقية لأشخاص طلبوا مساعدة نفسية، أجاب عنها مختصون، ومصنفة حسب نوع السؤال.")
    stats([("حالة في قاعدة المعرفة", d["knowledge_base"], "", False),
           ("استُبعدت لأنها عن الانتحار أو إيذاء النفس", d["excluded_crisis_cases"], "لا تُعرض كحالات مشابهة", False),
           ("سؤال لاختبار الرد", d["generation_test"], "مع رد مختص مرجعي", False),
           ("سؤال لاختبار التصنيف", d["classification_test"], "مع نوع السؤال", False)])

    section(2, "ماذا نخزّن لكل حالة", "ردود المختصين قصيرة (حوالي 180 حرفاً) فلا يفيد تقطيعها. بدلاً من ذلك قارنّا ما نخزّنه "
            "ونبحث فيه: سؤال الشخص، أو رد المختص، أو كليهما. المقياس: كم من الحالات المسترجعة تشترك مع السؤال في نوعه.")
    u = res["units"].copy()
    u["الوحدة"] = u["unit"].map(UNIT_AR)
    chart(u.melt(id_vars="الوحدة", value_vars=["dev_type_p@3", "test_type_p@1", "test_type_p@3"], var_name="m",
                 value_name="v").assign(m=lambda x: x["m"].map(COL_AR)), "الوحدة", "v", "m")
    st.markdown(f'<p class="fine">المختار: {UNIT_AR[res["chosen_unit"]]}. الاختيار العشوائي يعطي {base["type_precision"]:.3f}، '
                'لأن نوعي التشخيص والعلاج شائعان جداً في البيانات.</p>', unsafe_allow_html=True)

    section(3, "التضمين والبحث", "أصعب اختبار: نبحث بسؤال كل حالة بين جميع ردود المختصين. هل يأتي الرد الذي كُتب له أولاً؟ "
            "هذا يقيس هل يفهم النظام أن كلمات الشخص وكلمات المختص تتحدث عن الشيء نفسه.")
    rt = res["retrieval"].copy()
    best = rt.iloc[0]
    stats([("الرد الصحيح جاء أولاً", best["qa_match_hit@1"], f"عشوائياً: {1 / d['searchable_cases']:.3f}", True),
           ("الرد الصحيح ضمن أول 5", best["qa_match_hit@5"], f"عشوائياً: {5 / d['searchable_cases']:.3f}", True),
           ("MRR", best["qa_match_mrr"], f"أفضل طريقة: {best['method']}", True)])
    chart(rt.melt(id_vars="method", value_vars=["qa_match_hit@1", "qa_match_hit@5", "qa_match_mrr"], var_name="m",
                  value_name="v").assign(m=lambda x: x["m"].map(COL_AR)), "method", "v", "m", h=300)
    for name in (res.get("skipped_embedders") or {}):
        st.markdown(f'<div class="notice">لم يُقَس نموذج {esc(name)} في هذا التشغيل لأنه لم يُحمَّل. على جهاز متصل '
                    'بالإنترنت يُحمَّل مرة واحدة؛ أعد التقييم لتظهر نتيجته هنا.</div>', unsafe_allow_html=True)

    c = res["classification"]
    section(4, "فهم نوع السؤال", "تتعرف سكينة على نوع رسالتك (تشخيص، علاج، نمط حياة...) من الحالات المشابهة لها، "
            "وقارنّا ذلك بتصنيف البيانات لـ 150 سؤالاً.")
    stats([("Micro F1", c["micro_f1"], f"لو اخترنا «العلاج» دائماً: {c['majority_baseline_micro_f1']:.3f}", True),
           ("Macro F1", c["macro_f1"], "الأنواع النادرة تخفضه", True),
           ("كل الأنواع صحيحة", c["exact_set_accuracy"], "", True)])

    g = res["generation"]
    section(5, "صياغة الرد", f"على {g['n']} سؤال اختبار، قارنّا رد سكينة برد المختص الحقيقي على السؤال نفسه. "
            + ("الردود هنا من GPT." if g["mode"] == "llm" else "هذا التشغيل دون GPT، فالرد هو رد مختص على أقرب حالة."))
    stats([("تشابه الكلمات مع رد المختص (F1)", g["token_f1"], "", True), ("ROUGE-L", g["rougeL"], "", True),
           ("الأمانة للمصادر", g["faithfulness"], "كلمات الرد الموجودة في ردود المختصين", True),
           ("حُوّلت إلى مسار الأمان", g["crisis_routed"], "رسائل أزمة في الاختبار", False)])

    s = res["safety"]
    section(6, "الأمان", "قبل أي رد، تفحص سكينة الرسالة بحثاً عن أفكار انتحار أو إيذاء للنفس، وتعرض أرقام المساعدة بدل الرد العادي.")
    stats([("رسائل الأزمة التي كُشفت", s["probe_recall"], f"من {s['n_probes']} رسالة اختبار", True),
           ("دقة التنبيه", s["probe_precision"], "دون إنذارات خاطئة", True),
           ("رسالة أزمة في البيانات الحقيقية", s["dataset_flagged"], f"من {s['dataset_total']}", False)])

    st.markdown('<div class="honest"><b>قراءة صادقة.</b> البحث بالكلمات وحده يربط الشخص بالرد الصحيح أفضل كثيراً من العشوائي، '
                'لكنه يبقى ضعيفاً، لأن الشخص والمختص يعبّران بكلمات مختلفة. نموذج التضمين الدلالي e5 و GPT هما ما يُتوقع أن '
                'يسد هذه الفجوة؛ أعد التقييم على جهازك لقياس ذلك. رسائل اختبار الأمان كتبها المطوّر واستُخدمت أثناء كتابة '
                'القواعد، فهي فحص أولي لا اختبار مستقل.</div>', unsafe_allow_html=True)

