async function selectFolder() {
  const res = await fetch('/utils/select-folder');
  const { path } = await res.json();
  if (path) {
    alert(path);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('loadBtn').addEventListener('click', selectFolder);
})