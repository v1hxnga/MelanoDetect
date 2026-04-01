document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("fileInput");
  const previewWrapper = document.getElementById("previewWrapper");
  const imagePreview = document.getElementById("imagePreview");
  const fileName = document.getElementById("fileName");
  const uploadBox = document.getElementById("uploadBox");

  if (fileInput && imagePreview && previewWrapper && fileName) {
    fileInput.addEventListener("change", function () {
      const file = this.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function (e) {
        imagePreview.src = e.target.result;
        fileName.textContent = file.name;
        previewWrapper.classList.remove("hidden");
      };
      reader.readAsDataURL(file);
    });
  }

  if (uploadBox && fileInput) {
    uploadBox.addEventListener("dragover", function (e) {
      e.preventDefault();
      uploadBox.style.borderColor = "#22d3ee";
    });

    uploadBox.addEventListener("dragleave", function () {
      uploadBox.style.borderColor = "rgba(255,255,255,0.2)";
    });

    uploadBox.addEventListener("drop", function (e) {
      e.preventDefault();
      uploadBox.style.borderColor = "rgba(255,255,255,0.2)";
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        fileInput.files = files;
        fileInput.dispatchEvent(new Event("change"));
      }
    });
  }
});