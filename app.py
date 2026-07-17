import streamlit as st
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
}

.main-card{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    padding:40px;
    border-radius:25px;
    border:1px solid rgba(255,255,255,.15);
    box-shadow:0 10px 35px rgba(0,0,0,.45);
}

.title{
    font-size:42px;
    font-weight:700;
    text-align:center;
    color:white;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    margin-bottom:30px;
}

.stTextArea textarea{
    border-radius:15px;
    border:1px solid #4f46e5;
    font-size:17px;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:15px;
    border:none;
    background:linear-gradient(90deg,#4f46e5,#7c3aed);
    color:white;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.02);
    box-shadow:0 0 20px #7c3aed;
}

.footer{
    text-align:center;
    color:#94a3b8;
    margin-top:25px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# ---------------- UI ----------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title">📧 Email Spam Classifier</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Detect whether an email is <b>Spam</b> or <b>Not Spam</b> using Machine Learning.</div>',
    unsafe_allow_html=True
)

message = st.text_area(
    "Enter your Email",
    height=180,
    placeholder="Paste your email message here..."
)

if st.button("🔍 Analyze Email"):

    if message.strip()=="":

        st.warning("Please enter an email.")

    else:

        transformed = vectorizer.transform([message])

        prediction = model.predict(transformed)

        if prediction[0]==1:

            st.error("🚨 This Email is SPAM")

        else:

            st.success("✅ This Email is NOT SPAM")

st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit & Scikit-Learn</div>',
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)