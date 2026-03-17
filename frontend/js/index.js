async function selectFolder() {
  const res = await fetch('http://localhost:8000/utils/select-folder');
  const { path } = await res.json();
  if (path) {
    alert(path);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('loadBtn').addEventListener('click', selectFolder);
})