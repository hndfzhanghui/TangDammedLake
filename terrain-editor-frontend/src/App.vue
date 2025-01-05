<template>
  <v-app>
    <v-app-bar app>
      <v-toolbar-title>地形编辑器</v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn text @click="saveProject">
        保存项目
      </v-btn>
      <v-btn text @click="loadProject">
        加载项目
      </v-btn>
    </v-app-bar>

    <v-main>
      <terrain-editor></terrain-editor>
    </v-main>
  </v-app>
</template>

<script>
import TerrainEditor from './components/TerrainEditor.vue'

export default {
  name: 'App',
  components: {
    TerrainEditor
  },
  methods: {
    saveProject() {
      // TODO: 实现项目保存功能
    },
    loadProject() {
      // TODO: 实现项目加载功能
    }
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
