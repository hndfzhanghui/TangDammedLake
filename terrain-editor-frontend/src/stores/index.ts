import { defineStore } from 'pinia'

interface TerrainData {
  // Define terrain data structure
  elevation: number[][]
  coordinates: {
    lat: number
    lon: number
  }[][]
}

interface VisualizationSettings {
  showGrid: boolean
  showContours: boolean
  colorScheme: string
}

export const useMainStore = defineStore('main', {
  state: () => ({
    terrainData: null as TerrainData | null,
    visualizationSettings: {
      showGrid: true,
      showContours: true,
      colorScheme: 'terrain'
    } as VisualizationSettings
  }),
  actions: {
    setTerrainData(data: TerrainData) {
      this.terrainData = data
    },
    updateVisualizationSettings(settings: Partial<VisualizationSettings>) {
      this.visualizationSettings = {
        ...this.visualizationSettings,
        ...settings
      }
    }
  }
})
