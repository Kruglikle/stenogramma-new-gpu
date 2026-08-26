(() => {
  const openHistoryItem = (event) => {
    const item = event.target.closest("[data-history-url]");
    if (!item || event.target.closest("a")) return;
    window.location.href = item.dataset.historyUrl;
  };

  document.addEventListener("dblclick", openHistoryItem);
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    openHistoryItem(event);
  });

  document.addEventListener("submit", (event) => {
    const form = event.target.closest("[data-delete-form]");
    if (!form) return;

    const title = form.dataset.deleteTitle || "этот файл";
    if (!window.confirm(`Удалить "${title}"? Это освободит место, но восстановить результат будет нельзя.`)) {
      event.preventDefault();
    }
  });

  const setUploadProgress = (form, percent, label) => {
    const progress = form.querySelector("[data-upload-progress]");
    const bar = form.querySelector("[data-upload-progress-bar]");
    const percentText = form.querySelector("[data-upload-progress-percent]");
    const labelText = form.querySelector("[data-upload-progress-label]");
    if (!progress || !bar || !percentText || !labelText) return;

    progress.hidden = false;
    progress.classList.toggle("is-indeterminate", percent === null);
    if (percent === null) {
      percentText.textContent = "";
      bar.style.width = "";
    } else {
      const normalized = Math.max(0, Math.min(100, percent));
      percentText.textContent = `${normalized > 0 && normalized < 1 ? normalized.toFixed(1) : Math.round(normalized)}%`;
      bar.style.width = `${normalized}%`;
    }
    labelText.textContent = label;
  };

  const formatBytes = (bytes) => {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 Б";
    const units = ["Б", "КБ", "МБ", "ГБ"];
    const unitIndex = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / (1024 ** unitIndex);
    return `${value >= 10 || unitIndex === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
  };

  const resetUploadForm = (form, message) => {
    form.dataset.submitting = "false";
    setUploadSubmitting(form, false);
    setUploadProgress(form, 0, message);
  };

  const setUploadSubmitting = (form, isSubmitting) => {
    const submitButton = form.querySelector("[data-upload-submit]");
    for (const element of form.elements) {
      if (element === submitButton) continue;
      element.disabled = isSubmitting;
    }
    if (submitButton) {
      submitButton.disabled = isSubmitting;
      submitButton.textContent = isSubmitting ? "Загружаю..." : "Запустить обработку";
    }
  };

  document.addEventListener("submit", (event) => {
    const form = event.target.closest("[data-upload-form]");
    if (!form) return;

    event.preventDefault();
    if (form.dataset.submitting === "true") return;

    const fileInput = form.querySelector('input[type="file"]');
    const selectedFile = fileInput?.files?.[0] || null;
    const maxUploadBytes = Number(form.dataset.maxUploadBytes || 0);
    if (selectedFile && maxUploadBytes > 0 && selectedFile.size > maxUploadBytes) {
      setUploadProgress(
        form,
        0,
        `Файл ${formatBytes(selectedFile.size)} превышает лимит ${formatBytes(maxUploadBytes)}.`,
      );
      return;
    }

    form.dataset.submitting = "true";

    // FormData must be created before disabling fields: disabled fields are not submitted.
    const formData = new FormData(form);
    const hasFile = Boolean(selectedFile);
    setUploadSubmitting(form, true);
    setUploadProgress(
      form,
      hasFile ? 0 : null,
      hasFile ? `Подготовка файла ${formatBytes(selectedFile.size)}...` : "Отправляю задачу...",
    );

    const xhr = new XMLHttpRequest();
    xhr.open(form.method || "POST", form.action);
    xhr.upload.addEventListener("progress", (progressEvent) => {
      if (!progressEvent.lengthComputable) {
        setUploadProgress(form, null, `Загружено ${formatBytes(progressEvent.loaded)}...`);
        return;
      }
      setUploadProgress(
        form,
        (progressEvent.loaded / progressEvent.total) * 100,
        `Загружено ${formatBytes(progressEvent.loaded)} из ${formatBytes(progressEvent.total)}`,
      );
    });
    xhr.upload.addEventListener("load", () => {
      setUploadProgress(form, 100, "Файл передан. Создаю задачу...");
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 400) {
        setUploadProgress(form, 100, "Загрузка завершена. Открываю задачу...");
        window.location.href = xhr.responseURL || "/ui/upload";
        return;
      }

      const responseDocument = new DOMParser().parseFromString(xhr.responseText || "", "text/html");
      const serverMessage = responseDocument.querySelector(".error")?.textContent?.trim();
      const fallbackMessage = xhr.status === 413
        ? "Файл превышает допустимый размер."
        : `Сервер отклонил загрузку (код ${xhr.status}).`;
      resetUploadForm(form, serverMessage || fallbackMessage);
    });
    xhr.addEventListener("error", () => {
      resetUploadForm(form, "Соединение прервано. Обновите страницу и повторите загрузку.");
    });
    xhr.addEventListener("abort", () => {
      resetUploadForm(form, "Загрузка отменена.");
    });
    xhr.send(formData);
  });
})();
