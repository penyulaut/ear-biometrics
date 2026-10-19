const $ = (selector) => document.querySelector(selector);
let selectedFile = null;
const uploadView = $("#upload-view");
const resultView = $("#result-view");
const identityView = $("#identity-view");

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) return showNotice("File harus berupa gambar JPG, JPEG, PNG, atau WEBP.", true);
  if (file.size > 10 * 1024 * 1024) return showNotice("Ukuran gambar maksimal 10 MB.", true);
  selectedFile = file;
  $("#file-status").innerHTML = `<i class="bi bi-file-earmark-image"></i> ${file.name}`;
  $("#identify-btn").disabled = !$("#consent").checked;
  showNotice("Foto siap dianalisis. Centang persetujuan untuk melanjutkan.", false);
}
function showNotice(message, error) {
  const notice = $("#notice");
  notice.textContent = message;
  notice.classList.remove("hidden");
  notice.style.background = error ? "#fff0ef" : "";
  notice.style.color = error ? "#a94b47" : "";
}
function displayResult(data, previewUrl) {
  $("#upload-view").classList.add("hidden");
  $("#result-view").classList.remove("hidden");
  $("#page-title").textContent = "Hasil identifikasi";
  $("#page-subtitle").textContent = "Tinjau kecocokan identitas dari foto telinga yang Anda unggah.";
  $("#state-pill").innerHTML = '<i class="bi bi-check-circle-fill"></i> Identifikasi selesai · demo';
  $("#preview").src = previewUrl;
  $("#result-filename").textContent = data.filename;
  $("#result-name").textContent = data.name;
  $("#result-id").textContent = data.identity_id;
  $("#person-avatar").textContent = data.name.split(" ").map((part) => part[0]).join("").slice(0, 2);
  $("#confidence").textContent = `${String(data.confidence).replace(".", ",")}%`;
  $("#confidence-bar").style.width = `${Math.min(data.confidence, 100)}%`;
  $("#notice").classList.add("hidden");
  document.querySelectorAll(".step")[0].classList.remove("active");
  $("#step-identify").classList.add("done");
  $("#step-result").classList.add("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}
async function identify() {
  if (!selectedFile) return;
  $("#identify-btn").disabled = true;
  $("#identify-btn").textContent = "Memproses foto…";
  const previewUrl = URL.createObjectURL(selectedFile);
  try {
    const form = new FormData();
    form.append("file", selectedFile);
    const response = await fetch("/api/identify", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Identifikasi gagal.");
    displayResult(data, previewUrl);
  } catch (error) {
    showNotice(error.message, true);
    $("#identify-btn").disabled = false;
  } finally {
    $("#identify-btn").innerHTML = '<i class="bi bi-search"></i> Identifikasi telinga';
  }
}
$("#choose-btn").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", (event) => setFile(event.target.files[0]));
$("#consent").addEventListener("change", (event) => { $("#identify-btn").disabled = !selectedFile || !event.target.checked; });
$("#identify-btn").addEventListener("click", identify);
$("#dropzone").addEventListener("dragover", (event) => { event.preventDefault(); $("#dropzone").style.borderColor = "var(--teal)"; });
$("#dropzone").addEventListener("dragleave", () => { $("#dropzone").style.borderColor = ""; });
$("#dropzone").addEventListener("drop", (event) => { event.preventDefault(); setFile(event.dataTransfer.files[0]); });
$("#another-btn").addEventListener("click", () => window.location.reload());
$("#camera-btn").addEventListener("click", () => showNotice("Pengambilan kamera akan tersedia saat perangkat memberikan izin kamera.", false));

function showView(viewName) {
  uploadView.classList.toggle("hidden", viewName !== "upload");
  resultView.classList.toggle("hidden", viewName !== "result");
  identityView.classList.toggle("hidden", viewName !== "identity");
  $(".privacy").classList.toggle("hidden", viewName === "identity");

  if (viewName === "identity") {
    $("#page-title").textContent = "Data identitas";
    $("#page-subtitle").textContent = "Daftar orang yang sudah terdaftar untuk proses identifikasi.";
    $(".breadcrumb").textContent = "Ruang kerja　›　Data identitas";
    $("#state-pill").innerHTML = '<i class="bi bi-people"></i> Data terdaftar';
  } else if (viewName === "upload") {
    $("#page-title").textContent = "Identifikasi biometrik telinga";
    $("#page-subtitle").textContent = "Unggah foto telinga, lalu cocokkan dengan identitas yang sudah terdaftar.";
    $(".breadcrumb").textContent = "Ruang kerja　›　Identifikasi telinga";
    $("#state-pill").innerHTML = '<i class="bi bi-circle-fill"></i> Menunggu gambar';
  }
}

async function loadIdentities() {
  const response = await fetch("/api/identities");
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Data identitas gagal dimuat.");

  $("#identity-count").textContent = `${data.identities.length} orang`;
  $("#identity-table-body").replaceChildren(
    ...data.identities.map((identity, index) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${index + 1}</td>
        <td><div class="identity-name"><span class="person-avatar">${identity.initials}</span><strong></strong></div></td>
        <td>${identity.identity_id}</td>
        <td><span class="status-badge"><i class="bi bi-check-circle-fill"></i>${identity.status}</span></td>
      `;
      row.querySelector("strong").textContent = identity.name;
      return row;
    }),
  );
}

document.querySelector(".sidebar nav").addEventListener("click", async (event) => {
  const item = event.target.closest(".nav-item");
  if (!item) return;
  event.preventDefault();

  document.querySelectorAll(".nav-item").forEach((navItem) => navItem.classList.remove("active"));
  item.classList.add("active");
  if (item.dataset.view === "identity") {
      showView("identity");
      window.history.pushState({}, "", "/identitas");
      try {
        await loadIdentities();
      } catch (error) {
        showNotice(error.message, true);
      }
    return;
  }
  showView("upload");
  window.history.pushState({}, "", "/");
});

if (window.location.pathname === "/identitas" || window.location.pathname === "/identitas/") {
  const identityNavItem = document.querySelector('.nav-item[data-view="identity"]');
  identityNavItem.classList.add("active");
  document.querySelector('.nav-item[data-view="upload"]').classList.remove("active");
  showView("identity");
  loadIdentities().catch((error) => showNotice(error.message, true));
}
