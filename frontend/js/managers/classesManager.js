// Classes Manager Module
class ClassesManager {
  constructor() {
    this.classes = [];
    this.selectedClassId = null;
    this.datasetId = null;
  }

  async init(datasetId) {
    this.datasetId = datasetId;
    await this.loadClasses();
    this.renderClassSelect();
  }

  async loadClasses() {
    try {
      const res = await fetch(`http://localhost:8000/api/datasets/${this.datasetId}/classes`);
      if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
      this.classes = await res.json();
      
      if (this.classes.length > 0) {
        this.selectedClassId = this.classes[0].id;
      }
    } catch (err) {
      console.error('Failed to load classes:', err);
      this.classes = [];
    }
  }

  renderClassSelect() {
    const select = document.getElementById('class-select');
    if (!select) return;

    select.innerHTML = '';
    
    if (this.classes.length === 0) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = 'Нет классов';
      select.appendChild(option);
      return;
    }

    this.classes.forEach(cls => {
      const option = document.createElement('option');
      option.value = cls.id;
      option.textContent = cls.name;
      if (cls.id === this.selectedClassId) {
        option.selected = true;
      }
      select.appendChild(option);
    });

    select.addEventListener('change', (e) => {
      this.selectedClassId = parseInt(e.target.value);
    });
  }

  getSelectedClass() {
    return this.classes.find(cls => cls.id === this.selectedClassId);
  }

  async addClass(name, color) {
    try {
      const res = await fetch(`http://localhost:8000/api/datasets/${this.datasetId}/classes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, color })
      });

      if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
      
      const newClass = await res.json();
      this.classes.push(newClass);
      this.renderClassSelect();
      return newClass;
    } catch (err) {
      console.error('Failed to add class:', err);
      throw err;
    }
  }

  async removeClass(classId) {
    try {
      const res = await fetch(`http://localhost:8000/api/datasets/${this.datasetId}/classes/${classId}`, {
        method: 'DELETE'
      });

      if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
      
      this.classes = this.classes.filter(cls => cls.id !== classId);
      
      if (this.selectedClassId === classId && this.classes.length > 0) {
        this.selectedClassId = this.classes[0].id;
      }
      
      this.renderClassSelect();
    } catch (err) {
      console.error('Failed to remove class:', err);
      throw err;
    }
  }

  async updateClass(classId, updates) {
    try {
      const res = await fetch(`http://localhost:8000/api/datasets/${this.datasetId}/classes/${classId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      });

      if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
      
      const updatedClass = await res.json();
      const index = this.classes.findIndex(cls => cls.id === classId);
      if (index !== -1) {
        this.classes[index] = updatedClass;
      }
      
      this.renderClassSelect();
      return updatedClass;
    } catch (err) {
      console.error('Failed to update class:', err);
      throw err;
    }
  }
}

export const classesManager = new ClassesManager();
