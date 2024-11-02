
document.getElementById("uploadForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    const imageInput = document.getElementById("imageInput");
    const resultDiv = document.getElementById("result");
    const loadingDiv = document.querySelector(".loading");
    const submitBtn = document.querySelector(".submit-btn");

    if (imageInput.files.length === 0) {
        resultDiv.textContent = "Please select an image first.";
        resultDiv.style.display = "block";
        return;
    }

    // Show loading animation
    loadingDiv.style.display = "block";
    submitBtn.disabled = true;
    resultDiv.style.display = "none";

    const formData = new FormData();
    formData.append("file", imageInput.files[0]);

    try {
        const response = await fetch("/scorefile/", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.error) {
            resultDiv.textContent = `Error: ${data.error}`;
        } else {
            resultDiv.textContent = `${data.predicted_class}`;
        }
    } catch (error) {
        console.error('Error:', error);
        resultDiv.textContent = `Error: ${error.message}`;
    } finally {
        loadingDiv.style.display = "none";
        submitBtn.disabled = false;
        resultDiv.style.display = "block";
    }
});

document.getElementById("imageInput").addEventListener("change", function (event) {
    const preview = document.getElementById("preview");
    const submitBtn = document.querySelector(".submit-btn");
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.src = e.target.result;
            preview.style.display = "block";
            submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    } else {
        preview.style.display = "none";
        submitBtn.disabled = true;
    }
});

const dropZone = document.querySelector('.file-input');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#2B6CB0';
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#4299E1';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#4299E1';

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        const input = document.getElementById('imageInput');
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event('change'));
    }
});