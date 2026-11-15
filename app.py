
import streamlit as st
from predict_email import predict_email

# Page configuration
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
}
.stButton button {
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# Clear input callback
def clear_email():
    st.session_state.email_input = ""


# Sidebar
with st.sidebar:
    st.title("📧 About")
    st.write("Spam Email Detection System")
    st.divider()

    st.subheader("Machine Learning")
    st.write("Algorithm: Linear SVM")
    st.write("Text processing: TF-IDF")
    st.write("Training emails: 24,109")

    st.divider()
    st.caption("Built with Python and Streamlit")


# Main heading
st.markdown(
    '<p class="title">📧 Spam Email Detector</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Identify suspicious emails using Machine Learning.</p>',
    unsafe_allow_html=True
)

st.divider()


# Email input
st.subheader("Analyze an Email")

email_text = st.text_area(
    "Paste your email content:",
    height=220,
    placeholder="Paste the email you want to check...",
    key="email_input"
)

# Character counter
st.caption(f"Characters entered: {len(email_text)}")


# Buttons
col1, col2 = st.columns(2)

with col1:
    check_button = st.button(
        "🔍 Check Email",
        use_container_width=True
    )

with col2:
    st.button(
        "🗑️ Clear Email",
        use_container_width=True,
        on_click=clear_email
    )


# Prediction
if check_button:

    if not email_text.strip():
        st.warning("Please enter an email first.")

    else:
        result = predict_email(email_text)

        if result is None:
            st.warning("Unable to analyze this email.")

        else:
            prediction = result["prediction"].lower()
            confidence = result["confidence"]

            st.divider()
            st.subheader("Prediction Result")

            if prediction == "spam":
                st.error("🚨 This email is SPAM!")

            elif prediction == "ham":
                st.success("✅ This email is LEGITIMATE.")

            else:
                st.warning("Unexpected prediction returned by the model.")

            if prediction in ["spam", "ham"]:
                st.metric(
                    "Model-estimated probability for this prediction",
                    f"{confidence:.2f}%"
                )


# Footer
st.divider()

st.caption(
    "Spam Email Detector | Linear SVM + TF-IDF"
)
