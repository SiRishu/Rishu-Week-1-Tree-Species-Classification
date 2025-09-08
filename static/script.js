document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("file");
  const resultBox = document.getElementById("result");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    // check if file selected
    if (!fileInput.files.length) {
      resultBox.textContent = "⚠️ Please select an image first.";
      resultBox.style.background = "#ffebee";
      resultBox.style.color = "#c62828";
      return;
    }

    // show loading state
    resultBox.textContent = "⏳ Predicting... please wait.";
    resultBox.style.background = "#e3f2fd";
    resultBox.style.color = "#1565c0";

    // prepare form data
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Server error: " + response.statusText);
      }

      const data = await response.json();

      // success prediction
      resultBox.textContent = `🌳 Predicted Species: ${data.prediction}`;
      resultBox.style.background = "#e8f5e9";
      resultBox.style.color = "#2e7d32";
    } catch (error) {
      console.error(error);
      resultBox.textContent = "❌ Error: Could not get prediction.";
      resultBox.style.background = "#ffebee";
      resultBox.style.color = "#c62828";
    }
  });
});
