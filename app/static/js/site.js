document.addEventListener("DOMContentLoaded", () => {
  const header = document.getElementById("siteHeader");
  const fileInput = document.getElementById("fileInput");
  const previewBox = document.getElementById("previewBox");
  const previewImage = document.getElementById("previewImage");
  const previewName = document.getElementById("previewName");
  const uploadZone = document.getElementById("uploadZone");

  if (header) {
    const onScroll = () => {
      if (window.scrollY > 10) {
        header.classList.add("scrolled");
      } else {
        header.classList.remove("scrolled");
      }
    };
    window.addEventListener("scroll", onScroll);
    onScroll();
  }

  document.querySelectorAll(".tilt-card").forEach((card) => {
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      const rotateX = ((y - centerY) / centerY) * -3;
      const rotateY = ((x - centerX) / centerX) * 3;

      card.style.transform = `perspective(900px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-2px)`;
    });

    card.addEventListener("mouseleave", () => {
      card.style.transform = "perspective(900px) rotateX(0deg) rotateY(0deg)";
    });
  });

  if (fileInput && previewBox && previewImage && previewName) {
    fileInput.addEventListener("change", function () {
      const file = this.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function (e) {
        previewImage.src = e.target.result;
        previewName.textContent = file.name;
        previewBox.classList.remove("hidden");
      };
      reader.readAsDataURL(file);
    });
  }

  if (uploadZone && fileInput) {
    ["dragenter", "dragover"].forEach((eventName) => {
      uploadZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        uploadZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      uploadZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
      });
    });

    uploadZone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        fileInput.files = files;
        fileInput.dispatchEvent(new Event("change"));
      }
    });
  }

  document.querySelectorAll("[data-count]").forEach((counter) => {
    const target = Number(counter.dataset.count || 0);
    let current = 0;
    const step = Math.max(1, Math.ceil(target / 30));

    const updateCounter = () => {
      current += step;
      if (current >= target) {
        counter.textContent = target;
        return;
      }
      counter.textContent = current;
      requestAnimationFrame(updateCounter);
    };

    updateCounter();
  });
});

function clearImage() {
  const fileInput = document.getElementById("fileInput");
  const previewBox = document.getElementById("previewBox");
  const previewImage = document.getElementById("previewImage");
  const previewName = document.getElementById("previewName");

  if (fileInput) fileInput.value = "";
  if (previewImage) previewImage.src = "";
  if (previewName) previewName.textContent = "";
  if (previewBox) previewBox.classList.add("hidden");
}