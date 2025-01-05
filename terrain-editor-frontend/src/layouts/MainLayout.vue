<template>
  <div class="main-layout">
    <header class="header">
      <nav>
        <router-link to="/">Home</router-link>
        <router-link to="/editor">Terrain Editor</router-link>
      </nav>
    </header>
    
    <main class="content">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
// 基础布局组件
</script>

<style scoped>
.main-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.header {
  padding: 1rem;
  background: #2c3e50;
  color: white;
}

.header nav {
  display: flex;
  gap: 1rem;
}

.header a {
  color: white;
  text-decoration: none;
}

.content {
  flex: 1;
  padding: 1rem;
}
</style>
