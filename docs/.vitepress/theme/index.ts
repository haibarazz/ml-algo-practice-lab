import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { h } from 'vue'
import Giscus from '@giscus/vue'
import './custom.css'

const giscusConfig = {
  repo: import.meta.env.VITE_GISCUS_REPO,
  repoId: import.meta.env.VITE_GISCUS_REPO_ID,
  category: import.meta.env.VITE_GISCUS_CATEGORY,
  categoryId: import.meta.env.VITE_GISCUS_CATEGORY_ID
}

function hasGiscusConfig() {
  return Object.values(giscusConfig).every(Boolean)
}

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'doc-after': () =>
        hasGiscusConfig()
          ? h(Giscus, {
              repo: giscusConfig.repo,
              repoId: giscusConfig.repoId,
              category: giscusConfig.category,
              categoryId: giscusConfig.categoryId,
              mapping: 'pathname',
              strict: '0',
              reactionsEnabled: '1',
              emitMetadata: '0',
              inputPosition: 'top',
              theme: 'preferred_color_scheme',
              lang: 'zh-CN',
              loading: 'lazy'
            })
          : null
    })
  }
} satisfies Theme
