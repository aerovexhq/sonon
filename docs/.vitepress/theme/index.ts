import DefaultTheme from 'vitepress/theme'
import './custom.css'
import Playground from './components/Playground.vue'
import VoiceSynthesisStudio from './components/VoiceSynthesisStudio.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('Playground', Playground)
    app.component('VoiceSynthesisStudio', VoiceSynthesisStudio)
  }
}
