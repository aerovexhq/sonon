import DefaultTheme from 'vitepress/theme'
import './custom.css'
import Playground from './components/Playground.vue'
import VoiceSynthesisStudio from './components/VoiceSynthesisStudio.vue'
import LandingPage from './components/LandingPage.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }: { app: any }) {
    app.component('Playground', Playground)
    app.component('VoiceSynthesisStudio', VoiceSynthesisStudio)
    app.component('LandingPage', LandingPage)
  }
}
