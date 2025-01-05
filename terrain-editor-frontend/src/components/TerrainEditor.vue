<template>
  <v-container fluid>
    <!-- 初始化参数设置对话框 -->
    <v-dialog
      v-model="showInitDialog"
      persistent
      max-width="500"
    >
      <v-card>
        <v-card-title>
          初始化地形
          <v-spacer></v-spacer>
          <v-tooltip bottom>
            <template v-slot:activator="{ on, attrs }">
              <v-icon
                color="primary"
                v-bind="attrs"
                v-on="on"
              >
                mdi-help-circle
              </v-icon>
            </template>
            <span>请选择初始化方式</span>
          </v-tooltip>
        </v-card-title>

        <v-card-text>
          <!-- 选择初始化方式 -->
          <v-tabs v-model="initTab">
            <v-tab>
              <v-icon left>mdi-map-plus</v-icon>
              新建地图
            </v-tab>
            <v-tab>
              <v-icon left>mdi-file-import</v-icon>
              加载文件
            </v-tab>
          </v-tabs>

          <v-tabs-items v-model="initTab">
            <!-- 新建地图选项 -->
            <v-tab-item>
              <v-form ref="initForm" v-model="isInitFormValid" class="mt-4">
                <v-text-field
                  v-model.number="initParams.width"
                  label="地形宽度"
                  type="number"
                  :rules="[v => !!v || '请输入地形宽度', v => v > 0 || '宽度必须大于0']"
                  required
                ></v-text-field>

                <v-text-field
                  v-model.number="initParams.height"
                  label="地形长度"
                  type="number"
                  :rules="[v => !!v || '请输入地形长度', v => v > 0 || '长度必须大于0']"
                  required
                ></v-text-field>

                <v-text-field
                  v-model.number="initParams.gridSize"
                  label="网格大小"
                  type="number"
                  step="0.1"
                  :rules="[
                    v => !!v || '请输入网格大小',
                    v => v > 0 || '网格大小必须大于0',
                    v => v <= Math.min(initParams.width, initParams.height) / 10 || '网格太大'
                  ]"
                  required
                ></v-text-field>
              </v-form>
            </v-tab-item>

            <!-- 加载文件选项 -->
            <v-tab-item>
              <v-card flat class="mt-4">
                <v-card-text>
                  <v-file-input
                    v-model="initFile"
                    accept=".json"
                    label="选择地形文件"
                    prepend-icon="mdi-map"
                    :rules="[v => !!v || '请选择地形文件']"
                    show-size
                    truncate-length="25"
                  ></v-file-input>
                  <v-alert
                    type="info"
                    text
                    dense
                    class="mt-3"
                  >
                    <div class="text-body-2">
                      <v-icon small left>mdi-information</v-icon>
                      请选择之前保存的地形文件（JSON格式）
                    </div>
                  </v-alert>
                </v-card-text>
              </v-card>
            </v-tab-item>
          </v-tabs-items>
        </v-card-text>

        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn
            color="primary"
            :disabled="!canProceed"
            @click="handleInitConfirm"
          >
            确认并开始编辑
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- 修改参数警告对话框 -->
    <v-dialog
      v-model="showModifyWarning"
      max-width="400"
    >
      <v-card>
        <v-card-title class="error--text">
          警告
        </v-card-title>
        <v-card-text>
          修改地形基础参数将重建整个地形网格，当前的编辑数据可能会丢失。确定要继续吗？
        </v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn
            text
            @click="showModifyWarning = false"
          >
            取消
          </v-btn>
          <v-btn
            color="error"
            text
            @click="confirmModifyParams"
          >
            确定修改
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <v-row>
      <!-- 3D地形预览区域 -->
      <v-col cols="8">
        <canvas ref="terrainCanvas" class="terrain-canvas"></canvas>
      </v-col>
      
      <!-- 工具面板 -->
      <v-col cols="4">
        <v-card>
          <v-card-title class="d-flex align-center">
            地形编辑器
            <v-spacer></v-spacer>
            <v-tooltip bottom>
              <template v-slot:activator="{ on, attrs }">
                <v-btn icon @click="undo" :disabled="undoStack.length === 0" v-bind="attrs" v-on="on">
                  <v-icon>mdi-undo</v-icon>
                </v-btn>
              </template>
              <span>撤销 (Ctrl+Z)</span>
            </v-tooltip>
            <v-tooltip bottom>
              <template v-slot:activator="{ on, attrs }">
                <v-btn icon @click="redo" :disabled="redoStack.length === 0" v-bind="attrs" v-on="on">
                  <v-icon>mdi-redo</v-icon>
                </v-btn>
              </template>
              <span>重做 (Ctrl+Shift+Z)</span>
            </v-tooltip>
          </v-card-title>

          <!-- 添加鼠标位置信息显示 -->
          <v-card-text class="pb-0">
            <v-card outlined class="mb-4">
              <v-card-text class="py-2">
                <div class="d-flex align-center mb-1">
                  <v-icon small class="mr-2">mdi-cursor-default</v-icon>
                  <span class="text-subtitle-2">鼠标位置</span>
                </div>
                <div class="d-flex justify-space-between">
                  <span>X: {{ mouseWorldPos.x.toFixed(2) }}</span>
                  <span>Y: {{ mouseWorldPos.y.toFixed(2) }}</span>
                  <span>Z: {{ mouseWorldPos.z.toFixed(2) }}</span>
                </div>
              </v-card-text>
            </v-card>

            <!-- 添加高度图例 -->
            <v-card outlined class="mb-4">
              <v-card-text class="py-2">
                <div class="d-flex align-center mb-2">
                  <v-icon small class="mr-2">mdi-gradient-vertical</v-icon>
                  <span class="text-subtitle-2">高度图例</span>
                </div>
                <div class="height-scale mb-2">
                  <div class="height-gradient"></div>
                  <div class="d-flex justify-space-between">
                    <span>{{ minHeight.toFixed(1) }}m</span>
                    <span>{{ ((minHeight + maxHeight) / 2).toFixed(1) }}m</span>
                    <span>{{ maxHeight.toFixed(1) }}m</span>
                  </div>
                </div>
                <div class="height-legend">
                  <div v-for="(color, index) in heightColors" :key="index" class="d-flex align-center mb-1">
                    <div class="color-box mr-2" :style="{ backgroundColor: colorToHex(color) }"></div>
                    <span class="text-caption">{{ getHeightRangeText(index) }}</span>
                  </div>
                </div>
              </v-card-text>
            </v-card>
          </v-card-text>
          
          <v-card-text>
            <v-tabs v-model="activeTab">
              <v-tab>
                <v-icon left>mdi-pencil</v-icon>
                地形编辑
              </v-tab>
              <v-tab>
                <v-icon left>mdi-water</v-icon>
                水源设置
              </v-tab>
              <v-tab>
                <v-icon left>mdi-border-all</v-icon>
                边界条件
              </v-tab>
              <v-tab>
                <v-icon left>mdi-palette</v-icon>
                可视化
              </v-tab>
            </v-tabs>

            <v-tabs-items v-model="activeTab">
              <!-- 地形编辑面板 -->
              <v-tab-item>
                <v-list>
                  <!-- 编辑模式选择 -->
                  <v-list-item>
                    <v-btn-toggle
                      v-model="editMode"
                      mandatory
                      class="d-flex flex-wrap"
                    >
                      <v-btn
                        v-for="mode in editModes"
                        :key="mode.value"
                        :value="mode.value"
                        class="edit-mode-btn"
                      >
                        <v-icon left>{{ mode.icon }}</v-icon>
                        {{ mode.text }}
                      </v-btn>
                    </v-btn-toggle>
                  </v-list-item>

                  <!-- 工具参数 -->
                  <v-list-item>
                    <v-card flat width="100%">
                      <v-card-text>
                        <!-- 笔刷大小 -->
                        <div class="d-flex align-center mb-2">
                          <v-icon class="mr-2">mdi-circle-outline</v-icon>
                          <div class="text-subtitle-2">笔刷大小</div>
                          <v-spacer></v-spacer>
                          <div class="text-body-2">{{ editTools[editMode].size.toFixed(1) }}</div>
                        </div>
                        <v-slider
                          v-model="editTools[editMode].size"
                          :min="0.1"
                          :max="5"
                          step="0.1"
                          class="mt-0"
                        ></v-slider>

                        <!-- 工具强度 -->
                        <div class="d-flex align-center mb-2">
                          <v-icon class="mr-2">{{ editMode === 'smooth' ? 'mdi-blur' : 'mdi-arrow-up-down' }}</v-icon>
                          <div class="text-subtitle-2">{{ editMode === 'smooth' ? '平滑强度' : '修改强度' }}</div>
                          <v-spacer></v-spacer>
                          <div class="text-body-2">{{ editTools[editMode].strength.toFixed(2) }}</div>
                        </div>
                        <v-slider
                          v-model="editTools[editMode].strength"
                          :min="0.01"
                          :max="1"
                          step="0.01"
                          class="mt-0"
                        ></v-slider>

                        <!-- 平滑迭代次数（仅在平滑模式下显示） -->
                        <template v-if="editMode === 'smooth'">
                          <div class="d-flex align-center mb-2">
                            <v-icon class="mr-2">mdi-repeat</v-icon>
                            <div class="text-subtitle-2">迭代次数</div>
                            <v-spacer></v-spacer>
                            <div class="text-body-2">{{ editTools.smooth.iterations }}</div>
                          </div>
                          <v-slider
                            v-model="editTools.smooth.iterations"
                            :min="1"
                            :max="5"
                            step="1"
                            class="mt-0"
                          ></v-slider>
                        </template>
                      </v-card-text>
                    </v-card>
                  </v-list-item>

                  <!-- 操作提示 -->
                  <v-list-item v-if="editMode === 'height'">
                    <v-alert
                      type="info"
                      text
                      dense
                      class="mb-0"
                    >
                      <div class="d-flex align-center">
                        <v-icon left>mdi-information</v-icon>
                        按住Shift键可降低地形高度
                      </div>
                    </v-alert>
                  </v-list-item>
                </v-list>
              </v-tab-item>

              <!-- 地形参数设置面板 -->
              <v-tab-item>
                <v-list>
                  <v-list-item>
                    <v-text-field
                      v-model.number="terrainParams.width"
                      label="地形宽度"
                      type="number"
                      @change="updateTerrainGeometry"
                    ></v-text-field>
                  </v-list-item>
                  <v-list-item>
                    <v-text-field
                      v-model.number="terrainParams.height"
                      label="地形长度"
                      type="number"
                      @change="updateTerrainGeometry"
                    ></v-text-field>
                  </v-list-item>
                  <v-list-item>
                    <v-text-field
                      v-model.number="terrainParams.gridSize"
                      label="网格大小"
                      type="number"
                      step="0.1"
                      @change="updateTerrainGeometry"
                    ></v-text-field>
                  </v-list-item>
                  <v-list-item>
                    <v-btn color="primary" @click="resetTerrain">
                      重置地形
                    </v-btn>
                  </v-list-item>
                </v-list>
              </v-tab-item>

              <!-- 水源设置面板 -->
              <v-tab-item>
                <v-list>
                  <v-list-item v-for="(source, index) in waterSources" :key="index">
                    <v-text-field
                      v-model="source.flow"
                      label="流量"
                      type="number"
                    ></v-text-field>
                  </v-list-item>
                  <v-btn color="primary" @click="addWaterSource">
                    添加水源
                  </v-btn>
                </v-list>
              </v-tab-item>

              <!-- 边界条件面板 -->
              <v-tab-item>
                <v-list>
                  <v-list-item>
                    <v-select
                      v-model="boundaryType"
                      :items="boundaryTypes"
                      label="边界类型"
                    ></v-select>
                  </v-list-item>
                </v-list>
              </v-tab-item>

              <!-- 可视化设置面板 -->
              <v-tab-item>
                <v-list>
                  <v-list-item>
                    <v-select
                      v-model="colorMode"
                      :items="colorModes"
                      label="着色模式"
                      @change="updateTerrainColors"
                    ></v-select>
                  </v-list-item>
                  
                  <!-- 高度图例 -->
                  <v-list-item v-if="colorMode === 'height'">
                    <v-card flat width="100%">
                      <v-card-text>
                        <div class="d-flex align-center mb-2">
                          <div class="text-subtitle-2">高度范围：</div>
                          <v-spacer></v-spacer>
                          <div class="text-subtitle-2">{{ minHeight.toFixed(2) }} ~ {{ maxHeight.toFixed(2) }}</div>
                        </div>
                        
                        <!-- 高度图例 -->
                        <div class="height-legend">
                          <div 
                            v-for="(item, index) in legendColors" 
                            :key="index"
                            class="legend-item d-flex align-center mb-1"
                          >
                            <div 
                              class="legend-color mr-2" 
                              :style="{ backgroundColor: item.color }"
                            ></div>
                            <div class="legend-label">{{ item.label }}</div>
                          </div>
                        </div>
                      </v-card-text>
                    </v-card>
                  </v-list-item>
                </v-list>
              </v-tab-item>
            </v-tabs-items>
          </v-card-text>

          <v-card-actions class="d-flex flex-wrap justify-space-between">
            <v-btn 
              color="primary" 
              @click="saveProject"
              :loading="isSaving"
            >
              <v-icon left>mdi-content-save</v-icon>
              保存项目
            </v-btn>
            
            <v-btn 
              color="primary"
              :loading="isLoading"
            >
              <label style="cursor: pointer">
                <v-icon left>mdi-folder-open</v-icon>
                加载项目
                <input
                  type="file"
                  accept=".json"
                  style="display: none"
                  @change="loadProject"
                >
              </label>
            </v-btn>

            <v-btn 
              color="error" 
              @click="confirmReset"
            >
              <v-icon left>mdi-refresh</v-icon>
              重置地形
            </v-btn>
          </v-card-actions>

          <!-- 确认对话框 -->
          <v-dialog v-model="showResetConfirm" max-width="300">
            <v-card>
              <v-card-title>确认重置</v-card-title>
              <v-card-text>
                确定要重置地形吗？此操作不可撤销。
              </v-card-text>
              <v-card-actions>
                <v-spacer></v-spacer>
                <v-btn text @click="showResetConfirm = false">取消</v-btn>
                <v-btn color="error" text @click="confirmResetTerrain">确定</v-btn>
              </v-card-actions>
            </v-card>
          </v-dialog>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export default {
  name: 'TerrainEditor',
  data() {
    return {
      activeTab: 0,
      editMode: 'height',
      editModes: [
        { text: '抬升/降低', value: 'height', icon: 'mdi-arrow-up-down' },
        { text: '平滑地形', value: 'smooth', icon: 'mdi-blur' }
      ],
      brushSize: 1,
      brushStrength: 0.1,
      waterSources: [],
      boundaryType: 'fixed',
      boundaryTypes: [
        { text: '固定边界', value: 'fixed' },
        { text: '开放边界', value: 'open' },
        { text: '周期性边界', value: 'periodic' }
      ],
      isEditing: false,
      // 撤销/重做相关
      undoStack: [],
      redoStack: [],
      maxStackSize: 50,
      // 颜色可视化相关
      colorMode: 'height',
      colorModes: [
        { text: '高度着色', value: 'height' },
        { text: '坡度着色', value: 'slope' }
      ],
      minHeight: 0,
      maxHeight: 0,
      // 地形参数
      terrainParams: {
        width: 20,
        height: 20,
        gridSize: 0.2,  // 每个网格的大小
        startX: -10,
        startY: -10,
        endX: 10,
        endY: 10
      },
      
      // 编辑工具参数
      editTools: {
        height: {
          size: 1,
          strength: 0.1,
          showBrush: true
        },
        smooth: {
          size: 2,
          strength: 0.3,
          iterations: 1
        }
      },
      
      // 状态标志
      isSaving: false,
      isLoading: false,
      showResetConfirm: false,
      
      // 高度图例
      legendColors: [
        { color: '#FFFFFF', label: '山顶' },
        { color: '#8B4513', label: '山地' },
        { color: '#DAA520', label: '丘陵' },
        { color: '#228B22', label: '平地' },
        { color: '#00008B', label: '水面' }
      ],
      // 添加新的数据属性
      showInitDialog: true,
      showModifyWarning: false,
      isInitFormValid: false,
      isParamsLocked: false,
      initParams: {
        width: 100,
        height: 100,
        gridSize: 1
      },
      initTab: 0,
      initFile: null,
      mouseWorldPos: { x: 0, y: 0, z: 0 },
      heightColors: [
        { r: 1.0, g: 1.0, b: 1.0 }, // 白色（山顶）
        { r: 0.5, g: 0.35, b: 0.2 }, // 棕色（山地）
        { r: 0.6, g: 0.6, b: 0.2 }, // 黄色（丘陵）
        { r: 0.2, g: 0.6, b: 0.2 }, // 绿色（平地）
        { r: 0.0, g: 0.2, b: 0.5 }  // 深蓝（水面）
      ],
    };
  },
  computed: {
    canProceed() {
      // 根据当前选项卡判断是否可以继续
      if (this.initTab === 0) {
        return this.isInitFormValid;
      } else {
        return !!this.initFile;
      }
    }
  },
  async mounted() {
    // 初始化三维场景
    this.initThreeJS();
    
    // 添加事件监听
    window.addEventListener('resize', this.onWindowResize);
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    
    this.$refs.terrainCanvas.addEventListener('mousedown', this.onMouseDown);
    this.$refs.terrainCanvas.addEventListener('mousemove', this.onMouseMove);
    this.$refs.terrainCanvas.addEventListener('mouseup', this.onMouseUp);
  },
  beforeUnmount() {
    window.removeEventListener('resize', this.onWindowResize);
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    this.$refs.terrainCanvas.removeEventListener('mousedown', this.onMouseDown);
    this.$refs.terrainCanvas.removeEventListener('mousemove', this.onMouseMove);
    this.$refs.terrainCanvas.removeEventListener('mouseup', this.onMouseUp);
    if (this.renderer) {
      this.renderer.dispose();
    }
  },
  methods: {
    async handleInitConfirm() {
      if (this.initTab === 0) {
        // 新建地图
        await this.confirmInit();
      } else {
        // 加载文件
        await this.loadTerrainFile();
      }
      // 关闭对话框
      this.showInitDialog = false;
      this.isParamsLocked = true;
    },

    initThreeJS() {
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0xf0f0f0);
      
      // 添加坐标轴
      const axesHelper = new THREE.AxesHelper(10);
      this.scene.add(axesHelper);
      
      // 添加网格辅助线
      const gridHelper = new THREE.GridHelper(20, 20);
      gridHelper.rotation.x = Math.PI / 2;
      this.scene.add(gridHelper);
      
      // 设置相机
      this.camera = new THREE.PerspectiveCamera(
        45,
        this.$refs.terrainCanvas.clientWidth / this.$refs.terrainCanvas.clientHeight,
        0.1,
        1000
      );
      this.camera.position.set(20, 20, 20);
      this.camera.lookAt(0, 0, 0);
      
      // 设置渲染器
      this.renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        canvas: this.$refs.terrainCanvas
      });
      this.renderer.setSize(
        this.$refs.terrainCanvas.clientWidth,
        this.$refs.terrainCanvas.clientHeight
      );
      this.renderer.setPixelRatio(window.devicePixelRatio);
      
      // 设置控制器
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.mouseButtons = {
        MIDDLE: THREE.MOUSE.ROTATE,
        RIGHT: THREE.MOUSE.PAN
      };
      
      // 添加光源
      const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
      directionalLight.position.set(10, 10, 10);
      this.scene.add(directionalLight);
      
      const ambientLight = new THREE.AmbientLight(0x404040);
      this.scene.add(ambientLight);
      
      // 开始动画循环
      this.animate();
    },

    onWindowResize() {
      if (this.camera && this.renderer) {
        this.camera.aspect = this.$refs.terrainCanvas.clientWidth / this.$refs.terrainCanvas.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(
          this.$refs.terrainCanvas.clientWidth,
          this.$refs.terrainCanvas.clientHeight
        );
      }
    },
    
    initTerrain() {
      const { width, height, gridSize } = this.terrainParams;
      const segmentsX = Math.floor(width / gridSize);
      const segmentsY = Math.floor(height / gridSize);
      
      // 创建平坦的地形
      const geometry = new THREE.PlaneGeometry(
        width,
        height,
        segmentsX,
        segmentsY
      );
      
      const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        flatShading: true
      });
      
      // 初始化为平面（高度为0）
      const positions = geometry.attributes.position.array;
      for (let i = 0; i < positions.length; i += 3) {
        positions[i + 2] = 0;  // 设置z轴（高度）为0
      }
      
      // 如果已经存在地形，先移除它
      if (this.terrain) {
        this.scene.remove(this.terrain);
      }
      
      this.terrain = new THREE.Mesh(geometry, material);
      this.terrain.rotation.x = -Math.PI / 2;
      
      // 存储原始顶点位置
      this.originalVertices = geometry.attributes.position.array.slice();
      
      // 初始化顶点颜色
      const colors = new Float32Array(geometry.attributes.position.count * 3);
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      
      this.scene.add(this.terrain);
      
      // 创建射线投射器
      this.raycaster = new THREE.Raycaster();
      this.mouse = new THREE.Vector2();
      
      this.updateTerrainColors();
      
      // 添加笔刷指示器
      this.createBrushIndicator();
    },
    
    createBrushIndicator() {
      // 创建笔刷外圈（表示影响范围）
      const outerGeometry = new THREE.CircleGeometry(1, 32);
      const outerMaterial = new THREE.MeshBasicMaterial({
        color: 0xffff00,
        transparent: true,
        opacity: 0.2,
        side: THREE.DoubleSide,
        wireframe: true
      });
      this.brushIndicator = new THREE.Mesh(outerGeometry, outerMaterial);
      
      // 创建笔刷内圈（表示最大强度区域）
      const innerGeometry = new THREE.CircleGeometry(0.3, 32);
      const innerMaterial = new THREE.MeshBasicMaterial({
        color: 0xffff00,
        transparent: true,
        opacity: 0.4,
        side: THREE.DoubleSide
      });
      this.brushCoreIndicator = new THREE.Mesh(innerGeometry, innerMaterial);
      
      // 设置位置和方向
      this.brushIndicator.rotation.x = -Math.PI / 2;
      this.brushCoreIndicator.rotation.x = -Math.PI / 2;
      
      // 默认隐藏
      this.brushIndicator.visible = false;
      this.brushCoreIndicator.visible = false;
      
      // 添加到场景
      this.scene.add(this.brushIndicator);
      this.scene.add(this.brushCoreIndicator);
    },
    
    updateBrushIndicator() {
      // 检查所有必要的组件是否都已初始化
      if (!this.terrain || !this.camera || !this.mouse) return;
      
      // 确保射线投射器已初始化
      if (!this.raycaster) {
        this.raycaster = new THREE.Raycaster();
      }
      
      // 检查笔刷指示器是否存在
      if (!this.brushIndicator || !this.brushCoreIndicator) {
        this.createBrushIndicator();
        if (!this.brushIndicator || !this.brushCoreIndicator) return;
      }
      
      try {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObject(this.terrain);
        
        if (intersects.length > 0) {
          const point = intersects[0].point;
          const tool = this.editTools[this.editMode];
          
          // 更新外圈
          this.brushIndicator.position.copy(point);
          this.brushIndicator.scale.set(tool.size, tool.size, 1);
          this.brushIndicator.visible = true;
          
          // 更新内圈
          this.brushCoreIndicator.position.copy(point);
          this.brushCoreIndicator.scale.set(tool.size * 0.3, tool.size * 0.3, 1);
          this.brushCoreIndicator.visible = true;
          
          // 根据编辑模式更新颜色
          const color = this.editMode === 'smooth' ? 0x00ff00 : 0xffff00;
          this.brushIndicator.material.color.setHex(color);
          this.brushCoreIndicator.material.color.setHex(color);
        } else {
          this.brushIndicator.visible = false;
          this.brushCoreIndicator.visible = false;
        }
      } catch (error) {
        console.warn('更新笔刷指示器时发生错误:', error);
        // 出错时隐藏笔刷指示器
        if (this.brushIndicator) this.brushIndicator.visible = false;
        if (this.brushCoreIndicator) this.brushCoreIndicator.visible = false;
      }
    },
    
    animate() {
      if (!this.renderer) return;
      
      requestAnimationFrame(this.animate);
      if (this.controls) {
        this.controls.update();
      }
      this.renderer.render(this.scene, this.camera);
    },
    
    addWaterSource() {
      this.waterSources.push({
        flow: 0,
        position: { x: 0, y: 0 }
      });
    },

    onMouseDown(event) {
      // 只有左键才能编辑地形
      if (event.button === 0) {  // 0 表示左键
        this.isEditing = true;
        this.updateMousePosition(event);
        this.editTerrain(event);
      }
    },

    onMouseMove(event) {
      // 更新鼠标位置（无论是否在编辑状态）
      this.updateMousePosition(event);
      
      // 只有在左键按下时才编辑地形
      if (this.isEditing && event.buttons === 1) {
        this.editTerrain(event);
      }
    },

    onMouseUp(event) {
      if (event.button === 0) {  // 只处理左键释放
        this.isEditing = false;
      }
    },

    updateMousePosition(event) {
      if (!this.$refs.terrainCanvas) return;
      
      const rect = this.$refs.terrainCanvas.getBoundingClientRect();
      if (!rect) return;
      
      // 计算鼠标在画布中的归一化坐标（-1 到 1）
      if (!this.mouse) {
        this.mouse = new THREE.Vector2();
      }
      
      this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      
      // 更新世界坐标位置
      if (this.terrain && this.raycaster) {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObject(this.terrain);
        
        if (intersects.length > 0) {
          const point = intersects[0].point;
          // 由于地形是绕X轴旋转-90度的，我们需要调整坐标映射：
          // Three.js的y坐标对应我们想要的z（高度）
          // Three.js的z坐标对应我们想要的y（南北方向）
          this.mouseWorldPos = {
            x: point.x,                    // x保持不变（东西方向）
            y: -point.z,                   // Three.js的z对应y（南北方向，取反使正方向朝北）
            z: point.y                     // Three.js的y对应z（高度）
          };
        }
      }
      
      // 更新笔刷预览
      if (this.brushIndicator && this.brushCoreIndicator) {
        this.updateBrushIndicator();
      }
    },

    // 撤销/重做功能
    saveState() {
      // 确保在开始编辑时保存状态
      if (this.undoStack.length === 0 || 
          !this.compareArrays(this.terrain.geometry.attributes.position.array, this.undoStack[this.undoStack.length - 1])) {
        const positions = this.terrain.geometry.attributes.position.array.slice();
        this.undoStack.push(positions);
        if (this.undoStack.length > this.maxStackSize) {
          this.undoStack.shift();
        }
        this.redoStack = [];
      }
    },

    compareArrays(arr1, arr2) {
      if (!arr1 || !arr2 || arr1.length !== arr2.length) return false;
      for (let i = 0; i < arr1.length; i++) {
        if (Math.abs(arr1[i] - arr2[i]) > 0.0001) return false;
      }
      return true;
    },

    undo() {
      if (this.undoStack.length === 0) return;
      
      const currentState = this.terrain.geometry.attributes.position.array.slice();
      this.redoStack.push(currentState);
      
      const previousState = this.undoStack.pop();
      this.terrain.geometry.attributes.position.array.set(previousState);
      this.terrain.geometry.attributes.position.needsUpdate = true;
      this.terrain.geometry.computeVertexNormals();
      this.updateTerrainColors();
    },

    redo() {
      if (this.redoStack.length === 0) return;
      
      const currentState = this.terrain.geometry.attributes.position.array.slice();
      this.undoStack.push(currentState);
      
      const nextState = this.redoStack.pop();
      this.terrain.geometry.attributes.position.array.set(nextState);
      this.terrain.geometry.attributes.position.needsUpdate = true;
      this.terrain.geometry.computeVertexNormals();
      this.updateTerrainColors();
    },

    // 键盘事件处理
    onKeyDown(event) {
      if ((event.ctrlKey || event.metaKey) && !event.altKey) {
        if (event.key.toLowerCase() === 'z') {
          event.preventDefault();
          if (event.shiftKey) {
            this.redo();
          } else {
            this.undo();
          }
        }
      }
    },

    // 颜色可视化
    updateTerrainColors() {
      if (!this.terrain || !this.terrain.geometry) return;  // 添加安全检查
      
      const geometry = this.terrain.geometry;
      const positions = geometry.attributes.position.array;
      const colors = geometry.attributes.color.array;
      
      // 计算高度范围
      this.minHeight = Infinity;
      this.maxHeight = -Infinity;
      for (let i = 0; i < positions.length; i += 3) {
        const height = positions[i + 2];
        this.minHeight = Math.min(this.minHeight, height);
        this.maxHeight = Math.max(this.maxHeight, height);
      }
      
      // 确保有高度差，避免除以零
      if (Math.abs(this.maxHeight - this.minHeight) < 0.0001) {
        this.maxHeight = this.minHeight + 0.0001;
      }
      
      for (let i = 0; i < positions.length; i += 3) {
        const height = positions[i + 2];
        const t = (height - this.minHeight) / (this.maxHeight - this.minHeight);
        const color = this.getHeightColor(Math.max(0, Math.min(1, t)));  // 确保 t 在 0-1 之间
        
        colors[i] = color.r;
        colors[i + 1] = color.g;
        colors[i + 2] = color.b;
      }
      
      geometry.attributes.color.needsUpdate = true;
    },

    getHeightColor(t) {
      // 使用更适合地形的颜色方案
      const colors = [
        { r: 0.0, g: 0.2, b: 0.5 }, // 深蓝（水面）
        { r: 0.2, g: 0.6, b: 0.2 }, // 绿色（平地）
        { r: 0.6, g: 0.6, b: 0.2 }, // 黄色（丘陵）
        { r: 0.5, g: 0.35, b: 0.2 }, // 棕色（山地）
        { r: 1.0, g: 1.0, b: 1.0 }  // 白色（山顶）
      ];
      
      const index = t * (colors.length - 1);
      const i = Math.floor(index);
      const f = index - i;
      
      if (i >= colors.length - 1) return colors[colors.length - 1];
      
      return {
        r: colors[i].r + (colors[i + 1].r - colors[i].r) * f,
        g: colors[i].g + (colors[i + 1].g - colors[i].g) * f,
        b: colors[i].b + (colors[i + 1].b - colors[i].b) * f
      };
    },

    calculateSlope(geometry, vertexIndex) {
      const positions = geometry.attributes.position.array;
      const x = vertexIndex % 100;
      const y = Math.floor(vertexIndex / 100);
      
      let slope = 0;
      if (x > 0 && x < 99 && y > 0 && y < 99) {
        const center = positions[vertexIndex * 3 + 1];
        const left = positions[(vertexIndex - 1) * 3 + 1];
        const right = positions[(vertexIndex + 1) * 3 + 1];
        const top = positions[(vertexIndex - 100) * 3 + 1];
        const bottom = positions[(vertexIndex + 100) * 3 + 1];
        
        const dx = Math.max(Math.abs(left - center), Math.abs(right - center));
        const dy = Math.max(Math.abs(top - center), Math.abs(bottom - center));
        slope = Math.sqrt(dx * dx + dy * dy);
      }
      
      return slope;
    },

    getSlopeColor(slope) {
      const maxSlope = 1.0;
      const t = Math.min(slope / maxSlope, 1.0);
      
      return {
        r: t,
        g: 1 - t,
        b: 0
      };
    },

    // 保存和加载功能
    saveProject() {
      const data = {
        terrain: Array.from(this.terrain.geometry.attributes.position.array),
        waterSources: this.waterSources,
        boundaryType: this.boundaryType
      };
      
      const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = 'terrain_project.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },

    loadProject(event) {
      const file = event.target.files[0];
      if (!file) return;
      
      try {
        this.isLoading = true;
        const reader = new FileReader();
        
        reader.onload = async (e) => {
          try {
            const data = JSON.parse(e.target.result);
            
            // 更新地形
            this.terrain.geometry.attributes.position.array.set(data.terrain);
            this.terrain.geometry.attributes.position.needsUpdate = true;
            this.terrain.geometry.computeVertexNormals();
            
            // 更新其他数据
            this.waterSources = data.waterSources;
            this.boundaryType = data.boundaryType;
            
            // 更新颜色
            this.updateTerrainColors();
            
            // 清空撤销/重做栈
            this.undoStack = [];
            this.redoStack = [];
          } catch (error) {
            console.error('加载项目失败:', error);
          } finally {
            this.isLoading = false;
          }
        };
        
        reader.readAsText(file);
      } catch (error) {
        console.error('读取文件失败:', error);
        this.isLoading = false;
      }
    },

    editTerrain(event) {
      if (!this.terrain || !this.terrain.geometry) {
        console.warn('地形未初始化');
        return;
      }
      
      if (!this.raycaster) {
        this.raycaster = new THREE.Raycaster();
      }
      
      if (!this.mouse) {
        console.warn('鼠标位置未初始化');
        return;
      }
      
      this.raycaster.setFromCamera(this.mouse, this.camera);
      const intersects = this.raycaster.intersectObject(this.terrain);
      
      if (intersects.length > 0) {
        const intersection = intersects[0];
        const geometry = this.terrain.geometry;
        const positions = geometry.attributes.position.array;
        
        // 获取交点的局部坐标
        const point = intersection.point.clone();
        
        // 确保地形矩阵已初始化
        if (!this.terrain.matrixWorld) {
          this.terrain.updateMatrixWorld(true);
        }
        
        // 计算局部坐标
        const matrixWorldInverse = this.terrain.matrixWorld.clone().invert();
        const localPoint = point.applyMatrix4(matrixWorldInverse);
        
        // 获取地形网格的尺寸信息
        const { width, height } = this.terrainParams;
        const halfWidth = width / 2;
        const halfHeight = height / 2;
        
        const tool = this.editTools[this.editMode];
        
        // 保存当前状态用于撤销
        if (!this.isEditing) {
          this.saveState();
          this.isEditing = true;
        }

        if (this.editMode === 'height') {
          // 遍历所有顶点
          for (let i = 0; i < positions.length; i += 3) {
            const vertex = {
              x: positions[i],
              y: positions[i + 1],
              z: positions[i + 2]
            };
            
            // 计算顶点到笔刷中心的距离
            const dx = vertex.x - localPoint.x;
            const dy = vertex.y - localPoint.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // 确保顶点在笔刷范围内且在地形边界内
            if (distance < tool.size && 
                vertex.x >= -halfWidth && 
                vertex.x <= halfWidth && 
                vertex.y >= -halfHeight && 
                vertex.y <= halfHeight) {
              
              const influence = 1 - (distance / tool.size);
              const strength = event.shiftKey ? -tool.strength : tool.strength;
              positions[i + 2] += strength * influence;
            }
          }
        } else if (this.editMode === 'smooth') {
          // 调用平滑函数
          this.smoothTerrain(localPoint, tool.size, tool.strength, 1, halfWidth, halfHeight);
        }
        
        // 更新几何体
        geometry.attributes.position.needsUpdate = true;
        geometry.computeVertexNormals();
        this.updateTerrainColors();
      }
      
      // 更新笔刷指示器
      this.updateBrushIndicator();
    },

    // 平滑功能
    smoothTerrain(center, radius, strength, iterations = 1, halfWidth, halfHeight) {
      const geometry = this.terrain.geometry;
      const positions = geometry.attributes.position.array;
      
      // 计算搜索范围（以实际距离为单位）
      const searchRadius = radius * 0.5; // 使用笔刷大小的一半作为搜索半径
      
      for (let iter = 0; iter < iterations; iter++) {
        for (let i = 0; i < positions.length; i += 3) {
          const vertex = {
            x: positions[i],
            y: positions[i + 1],
            z: positions[i + 2]
          };
          
          // 计算顶点到笔刷中心的距离
          const dx = vertex.x - center.x;
          const dy = vertex.y - center.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          // 确保顶点在笔刷范围内且在地形边界内
          if (distance < radius && 
              vertex.x >= -halfWidth && 
              vertex.x <= halfWidth && 
              vertex.y >= -halfHeight && 
              vertex.y <= halfHeight) {
            
            const influence = Math.pow(1 - (distance / radius), 2);  // 使用平方函数使平滑效果更自然
            let sum = 0;
            let weightSum = 0;
            
            // 在搜索半径内查找邻近顶点
            for (let j = 0; j < positions.length; j += 3) {
              const neighborVertex = {
                x: positions[j],
                y: positions[j + 1],
                z: positions[j + 2]
              };
              
              const neighborDx = neighborVertex.x - vertex.x;
              const neighborDy = neighborVertex.y - vertex.y;
              const neighborDistance = Math.sqrt(neighborDx * neighborDx + neighborDy * neighborDy);
              
              // 只考虑搜索半径内的顶点
              if (neighborDistance <= searchRadius) {
                // 使用高斯权重
                const weight = Math.exp(-(neighborDistance * neighborDistance) / (2 * searchRadius * searchRadius));
                sum += neighborVertex.z * weight;
                weightSum += weight;
              }
            }
            
            // 计算平滑后的高度
            if (weightSum > 0) {
              const smoothedHeight = sum / weightSum;
              // 根据影响力和强度混合原始高度和平滑后的高度
              positions[i + 2] = vertex.z * (1 - influence * strength) + smoothedHeight * (influence * strength);
            }
          }
        }
      }
      
      // 更新几何体
      geometry.attributes.position.needsUpdate = true;
      geometry.computeVertexNormals();
      this.updateTerrainColors();
    },

    async updateTerrainGeometry() {
      if (this.isParamsLocked) {
        this.showModifyWarning = true;
        return;
      }
      await this.rebuildTerrain();
    },

    async confirmModifyParams() {
      this.showModifyWarning = false;
      await this.rebuildTerrain();
    },

    async rebuildTerrain() {
      try {
        // 保存当前的地形数据
        const oldPositions = this.terrain ? Array.from(this.terrain.geometry.attributes.position.array) : null;
        const oldWidth = this.terrainParams.width;
        const oldHeight = this.terrainParams.height;
        const oldGridSize = this.terrainParams.gridSize;

        // 重新创建地形
        await this.initTerrain();

        // 如果有旧数据，尝试进行插值
        if (oldPositions) {
          this.interpolateTerrainData(oldPositions, oldWidth, oldHeight, oldGridSize);
        }

        // 更新颜色和渲染
        this.updateTerrainColors();
        this.terrain.geometry.computeVertexNormals();
        
        // 清空撤销/重做栈
        this.undoStack = [];
        this.redoStack = [];
      } catch (error) {
        console.error('重建地形失败:', error);
      }
    },

    interpolateTerrainData(oldPositions, oldWidth, oldHeight, oldGridSize) {
      const newPositions = this.terrain.geometry.attributes.position.array;
      const oldSegmentsX = Math.floor(oldWidth / oldGridSize);
      const oldSegmentsY = Math.floor(oldHeight / oldGridSize);
      
      // 遍历新地形的每个顶点
      for (let i = 0; i < newPositions.length; i += 3) {
        const x = newPositions[i];
        const y = newPositions[i + 1];
        
        // 将坐标转换为旧网格的索引
        const oldX = ((x + oldWidth / 2) / oldWidth) * oldSegmentsX;
        const oldY = ((y + oldHeight / 2) / oldHeight) * oldSegmentsY;
        
        // 找到最近的四个顶点进行双线性插值
        const x1 = Math.floor(oldX);
        const x2 = Math.min(x1 + 1, oldSegmentsX);
        const y1 = Math.floor(oldY);
        const y2 = Math.min(y1 + 1, oldSegmentsY);
        
        const fx = oldX - x1;
        const fy = oldY - y1;
        
        // 获取四个角点的高度
        const h11 = oldPositions[(y1 * (oldSegmentsX + 1) + x1) * 3 + 2];
        const h21 = oldPositions[(y1 * (oldSegmentsX + 1) + x2) * 3 + 2];
        const h12 = oldPositions[(y2 * (oldSegmentsX + 1) + x1) * 3 + 2];
        const h22 = oldPositions[(y2 * (oldSegmentsX + 1) + x2) * 3 + 2];
        
        // 双线性插值
        const height = 
          h11 * (1 - fx) * (1 - fy) +
          h21 * fx * (1 - fy) +
          h12 * (1 - fx) * fy +
          h22 * fx * fy;
        
        newPositions[i + 2] = height;
      }
      
      this.terrain.geometry.attributes.position.needsUpdate = true;
    },

    resetTerrain() {
      if (!this.terrain || !this.originalVertices) return;  // 添加安全检查
      
      const positions = this.terrain.geometry.attributes.position.array;
      positions.set(this.originalVertices);
      this.terrain.geometry.attributes.position.needsUpdate = true;
      this.terrain.geometry.computeVertexNormals();
      this.updateTerrainColors();
    },

    confirmReset() {
      this.showResetConfirm = true;
    },
    
    confirmResetTerrain() {
      this.resetTerrain();
      this.showResetConfirm = false;
    },

    async confirmInit() {
      if (this.$refs.initForm.validate()) {
        // 复制初始参数到地形参数
        this.terrainParams = {
          width: this.initParams.width,
          height: this.initParams.height,
          gridSize: this.initParams.gridSize,
          startX: -this.initParams.width / 2,
          startY: -this.initParams.height / 2,
          endX: this.initParams.width / 2,
          endY: this.initParams.height / 2
        };
        
        // 初始化地形
        await this.initTerrain();
        this.showInitDialog = false;
        this.isParamsLocked = true;
      }
    },

    async loadTerrainFromJSON() {
      try {
        // 从本地JSON文件加载地形数据
        const response = await fetch('./terrain_data.json');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const terrainData = await response.json();
        
        console.log('读取到的地形数据:', terrainData);
        
        // 更新地形参数，使用原始尺寸
        this.terrainParams = {
          width: terrainData.width,
          height: terrainData.height,
          gridSize: 1,  // 使用原始网格大小
          startX: -terrainData.width / 2,
          startY: -terrainData.height / 2,
          endX: terrainData.width / 2,
          endY: terrainData.height / 2
        };
        
        // 创建新的地形几何体
        const geometry = new THREE.PlaneGeometry(
          this.terrainParams.width,
          this.terrainParams.height,
          terrainData.width - 1,
          terrainData.height - 1
        );
        
        // 设置高度数据
        const positions = geometry.attributes.position.array;
        const heights = terrainData.heights;
        
        // 检查数据是否存在
        if (!heights || !Array.isArray(heights)) {
          throw new Error('高程数据无效');
        }
        
        // 遍历每个顶点并设置高度
        let vertexIndex = 0;
        for (let row = 0; row < terrainData.height; row++) {
          for (let col = 0; col < terrainData.width; col++) {
            const height = heights[row][col];
            positions[vertexIndex * 3 + 2] = height;  // 使用原始高度值
            vertexIndex++;
          }
        }
        
        // 更新地形网格
        if (this.terrain) {
          this.scene.remove(this.terrain);
        }
        
        const material = new THREE.MeshPhongMaterial({
          vertexColors: true,
          side: THREE.DoubleSide,
          flatShading: true
        });
        
        this.terrain = new THREE.Mesh(geometry, material);
        this.terrain.rotation.x = -Math.PI / 2;
        
        // 确保矩阵被正确初始化
        this.terrain.updateMatrix();
        this.terrain.updateMatrixWorld(true);
        
        // 存储原始顶点位置
        this.originalVertices = geometry.attributes.position.array.slice();
        
        // 初始化顶点颜色
        const colors = new Float32Array(geometry.attributes.position.count * 3);
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        // 计算法线
        geometry.computeVertexNormals();
        
        // 更新颜色（在添加到场景之前）
        this.updateTerrainColors();
        
        // 添加到场景
        this.scene.add(this.terrain);
        
        // 更新相机位置
        const maxHeight = Math.max(...heights.flat());
        const cameraDistance = Math.max(this.terrainParams.width, this.terrainParams.height);
        this.camera.position.set(
          cameraDistance,
          cameraDistance,
          maxHeight * 2
        );
        this.camera.lookAt(0, 0, 0);
        
        // 重新创建笔刷指示器
        this.createBrushIndicator();
        
        // 禁用初始化对话框
        this.showInitDialog = false;
        this.isParamsLocked = true;
        
        console.log('地形数据加载成功');
      } catch (error) {
        console.error('加载地形数据失败:', error);
        // 如果加载失败，初始化一个空地形
        this.initTerrain();
      }
    },

    async loadTerrainFile() {
      if (!this.initFile) return;
      
      try {
        const reader = new FileReader();
        
        reader.onload = async (e) => {
          try {
            const terrainData = JSON.parse(e.target.result);
            
            // 更新地形参数
            this.terrainParams = {
              width: terrainData.width,
              height: terrainData.height,
              gridSize: 1,
              startX: -terrainData.width / 2,
              startY: -terrainData.height / 2,
              endX: terrainData.width / 2,
              endY: terrainData.height / 2
            };
            
            // 创建新的地形几何体
            const geometry = new THREE.PlaneGeometry(
              this.terrainParams.width,
              this.terrainParams.height,
              terrainData.width - 1,
              terrainData.height - 1
            );
            
            // 设置高度数据
            const positions = geometry.attributes.position.array;
            const heights = terrainData.heights;
            
            if (!heights || !Array.isArray(heights)) {
              throw new Error('无效的高程数据');
            }
            
            // 遍历每个顶点并设置高度
            let vertexIndex = 0;
            for (let row = 0; row < terrainData.height; row++) {
              for (let col = 0; col < terrainData.width; col++) {
                const height = heights[row][col];
                positions[vertexIndex * 3 + 2] = height;
                vertexIndex++;
              }
            }
            
            // 更新地形网格
            if (this.terrain) {
              this.scene.remove(this.terrain);
            }
            
            const material = new THREE.MeshPhongMaterial({
              vertexColors: true,
              side: THREE.DoubleSide,
              flatShading: true
            });
            
            this.terrain = new THREE.Mesh(geometry, material);
            this.terrain.rotation.x = -Math.PI / 2;
            
            // 确保矩阵被正确初始化
            this.terrain.updateMatrix();
            this.terrain.updateMatrixWorld(true);
            
            // 存储原始顶点位置
            this.originalVertices = geometry.attributes.position.array.slice();
            
            // 初始化顶点颜色
            const colors = new Float32Array(geometry.attributes.position.count * 3);
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            
            // 计算法线
            geometry.computeVertexNormals();
            
            // 更新颜色（在添加到场景之前）
            this.updateTerrainColors();
            
            // 添加到场景
            this.scene.add(this.terrain);
            
            // 更新相机位置
            const maxHeight = Math.max(...heights.flat());
            const cameraDistance = Math.max(this.terrainParams.width, this.terrainParams.height);
            this.camera.position.set(
              cameraDistance,
              cameraDistance,
              maxHeight * 2
            );
            this.camera.lookAt(0, 0, 0);
            
            // 重新创建笔刷指示器
            this.createBrushIndicator();
            
            // 清空撤销/重做栈
            this.undoStack = [];
            this.redoStack = [];
            
            // 关闭初始化对话框
            this.showInitDialog = false;
            this.isParamsLocked = true;
            
            console.log('地形数据加载成功');
          } catch (error) {
            console.error('解析地形文件失败:', error);
          }
        };
        
        reader.readAsText(this.initFile);
      } catch (error) {
        console.error('读取文件失败:', error);
      }
    },

    async initTerrainFromData(terrainData) {
      // 更新地形参数
      this.terrainParams = {
        width: terrainData.width,
        height: terrainData.height,
        gridSize: 1,
        startX: -terrainData.width / 2,
        startY: -terrainData.height / 2,
        endX: terrainData.width / 2,
        endY: terrainData.height / 2
      };
      
      // 创建新的地形几何体
      const geometry = new THREE.PlaneGeometry(
        this.terrainParams.width,
        this.terrainParams.height,
        terrainData.width - 1,
        terrainData.height - 1
      );
      
      // 设置高度数据
      const positions = geometry.attributes.position.array;
      const heights = terrainData.heights;
      
      // 遍历每个顶点并设置高度
      let vertexIndex = 0;
      for (let row = 0; row < terrainData.height; row++) {
        for (let col = 0; col < terrainData.width; col++) {
          const height = heights[row][col];
          positions[vertexIndex * 3 + 2] = height;
          vertexIndex++;
        }
      }
      
      // 更新地形网格
      if (this.terrain) {
        this.scene.remove(this.terrain);
      }
      
      const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        flatShading: true
      });
      
      this.terrain = new THREE.Mesh(geometry, material);
      this.terrain.rotation.x = -Math.PI / 2;
      
      // 存储原始顶点位置
      this.originalVertices = geometry.attributes.position.array.slice();
      
      // 初始化顶点颜色
      const colors = new Float32Array(geometry.attributes.position.count * 3);
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      
      this.scene.add(this.terrain);
      
      // 更新相机位置
      const maxHeight = terrainData.max_height;
      const cameraDistance = Math.max(this.terrainParams.width, this.terrainParams.height);
      this.camera.position.set(
        cameraDistance,
        cameraDistance,
        maxHeight * 2
      );
      this.camera.lookAt(0, 0, 0);
      
      // 更新颜色
      this.updateTerrainColors();
      
      // 重新创建笔刷指示器
      this.createBrushIndicator();
    },

    colorToHex(color) {
      const r = Math.floor(color.r * 255);
      const g = Math.floor(color.g * 255);
      const b = Math.floor(color.b * 255);
      return `rgb(${r}, ${g}, ${b})`;
    },

    getHeightRangeText(index) {
      const ranges = [
        '山顶',
        '山地',
        '丘陵',
        '平地',
        '水面'
      ];
      const heightRange = this.maxHeight - this.minHeight;
      const height = this.maxHeight - (heightRange * (index / (ranges.length - 1)));
      return `${ranges[index]} (${height.toFixed(1)}m)`;
    },
  }
};
</script>

<style scoped>
.terrain-canvas {
  width: 100%;
  height: 600px;
  background-color: #f0f0f0;
}

.height-legend {
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 8px;
}

.legend-item {
  height: 24px;
}

.legend-color {
  width: 24px;
  height: 24px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.legend-label {
  font-size: 14px;
}

/* 编辑模式按钮样式 */
.edit-mode-btn {
  flex: 1;
  min-width: 120px;
  height: 40px;
  margin: 4px;
  text-transform: none;
}

.edit-mode-btn .v-icon {
  margin-right: 8px;
}

/* 按钮悬停效果 */
.v-btn {
  transition: transform 0.2s;
}

.v-btn:hover {
  transform: scale(1.05);
}

/* 工具面板样式 */
.v-list-item {
  padding: 8px 16px;
}

.v-slider {
  margin-top: 0;
}

/* 标签页样式 */
.v-tab {
  text-transform: none;
  min-width: 100px;
}

/* 提示框样式 */
.v-tooltip {
  font-size: 12px;
}

/* 参数显示样式 */
.text-subtitle-2 {
  color: rgba(0, 0, 0, 0.6);
}

.text-body-2 {
  color: rgba(0, 0, 0, 0.87);
  font-weight: 500;
}

/* 工具卡片样式 */
.v-card--flat {
  background-color: transparent !important;
}

.height-scale {
  width: 100%;
  padding: 4px 0;
}

.height-gradient {
  width: 100%;
  height: 20px;
  background: linear-gradient(to top,
    rgb(0, 51, 128),   /* 深蓝（水面） */
    rgb(51, 153, 51),  /* 绿色（平地） */
    rgb(153, 153, 51), /* 黄色（丘陵） */
    rgb(128, 89, 51),  /* 棕色（山地） */
    rgb(255, 255, 255) /* 白色（山顶） */
  );
  border-radius: 4px;
  margin-bottom: 4px;
}

.color-box {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  border: 1px solid rgba(0, 0, 0, 0.1);
}

.height-legend {
  margin-top: 8px;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  padding-top: 8px;
}
</style> 