<template>
  <div class="terrain-editor">
    <div class="editor-container">
      <div class="toolbar">
        <button @click="addTerrain">Add Terrain</button>
        <button @click="removeTerrain">Remove Terrain</button>
        <button @click="simulateWaterFlow">Simulate</button>
      </div>
      
      <div class="canvas-container">
        <canvas ref="terrainCanvas"></canvas>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const terrainCanvas = ref<HTMLCanvasElement | null>(null)

onMounted(() => {
  initCanvas()
})

function initCanvas() {
  const canvas = terrainCanvas.value
  if (!canvas) return
  
  canvas.width = 800
  canvas.height = 600
  const ctx = canvas.getContext('2d')
  if (ctx) {
    // 初始化画布
    ctx.fillStyle = '#f0f0f0'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }
}

function addTerrain() {
  // 添加地形逻辑
}

function removeTerrain() {
  // 移除地形逻辑
}

function simulateWaterFlow() {
  // 模拟水流逻辑
}
</script>

<style scoped>
.terrain-editor {
  height: 100%;
}

.editor-container {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.toolbar {
  padding: 1rem;
  background: #f5f5f5;
  border-bottom: 1px solid #ddd;
}

.toolbar button {
  margin-right: 0.5rem;
  padding: 0.5rem 1rem;
  background: #2c3e50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.canvas-container {
  flex: 1;
  padding: 1rem;
}

canvas {
  width: 100%;
  height: 100%;
  border: 1px solid #ddd;
  background: white;
}
</style>
