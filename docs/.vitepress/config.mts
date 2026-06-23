import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitepress'

const docsRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')

type SidebarItem = {
  text: string
  link?: string
  collapsed?: boolean
  items?: SidebarItem[]
}

const base = process.env.VITEPRESS_BASE ?? '/ml-algo-practice-lab/'

function readTitle(filePath: string, fallback: string): string {
  if (!existsSync(filePath)) return fallback
  const text = readFileSync(filePath, 'utf8')
  const match = text.match(/^#\s+(.+)$/m)
  return match ? match[1].trim() : fallback
}

function moduleNamesFromReadme(sectionDir: string): string[] {
  const readme = resolve(sectionDir, 'README.md')
  if (!existsSync(readme)) return []

  const text = readFileSync(readme, 'utf8')
  const linked = [...text.matchAll(/\]\(\.\/([^/)]+)\/\1\.md\)/g)].map((match) => match[1])
  if (linked.length > 0) return linked

  return [...text.matchAll(/^-\s+`([^`]+)`:/gm)].map((match) => match[1])
}

function sectionItems(rootDir: string, urlPrefix: string, section: string): SidebarItem[] {
  const sectionDir = resolve(docsRoot, rootDir, section)
  const moduleNames = moduleNamesFromReadme(sectionDir)

  if (moduleNames.length > 0) {
    return moduleNames.map((name) => {
      const page = resolve(sectionDir, name, `${name}.md`)
      return {
        text: readTitle(page, name),
        link: `${urlPrefix}/${section}/${name}/${name}`
      }
    })
  }

  if (!existsSync(sectionDir)) return []

  return readdirSync(sectionDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const page = resolve(sectionDir, entry.name, `${entry.name}.md`)
      return {
        text: readTitle(page, entry.name),
        link: `${urlPrefix}/${section}/${entry.name}/${entry.name}`
      }
    })
}

function makeGroup(rootDir: string, urlPrefix: string, section: string, text: string): SidebarItem {
  return {
    text,
    link: `${urlPrefix}/${section}/`,
    collapsed: true,
    items: [
      { text: '分组首页', link: `${urlPrefix}/${section}/` },
      ...sectionItems(rootDir, urlPrefix, section)
    ]
  }
}

const handwritingGroups = [
  ['00_math_primitives', 'Math / Primitive'],
  ['01_classical_ml', 'Classical ML'],
  ['02_dl_basics', 'Deep Learning Basics'],
  ['03_attention_transformer', 'Attention / Transformer'],
  ['04_llm_training_alignment', 'LLM Training / Alignment']
] as const

const minimindGroups = [
  ['00_project_map', '项目地图'],
  ['01_tokenizer_and_data', 'Tokenizer 与数据'],
  ['02_model_architecture', '模型结构'],
  ['03_pretraining', '预训练'],
  ['04_sft', 'SFT'],
  ['05_preference_alignment', '偏好对齐'],
  ['06_inference', '推理'],
  ['07_evaluation', '评估'],
  ['08_system_architecture', '系统架构']
] as const

const sidebar: SidebarItem[] = [
  {
    text: '介绍',
    items: [
      { text: '项目概览', link: '/' },
      { text: '使用指南', link: '/guide' }
    ]
  },
  {
    text: '机器学习/深度学习手撕',
    items: [
      { text: '主线说明', link: '/ml_dl_handwriting/' },
      { text: '模块索引', link: '/ml_dl_handwriting/MODULE_INDEX' },
      ...handwritingGroups.map(([section, text]) =>
        makeGroup('ml_dl_handwriting', '/ml_dl_handwriting', section, text)
      )
    ]
  },
  {
    text: 'MiniMind 项目拆解',
    items: [
      { text: '项目说明', link: '/projects/minimind/' },
      { text: '模块索引', link: '/projects/minimind/MODULE_INDEX' },
      { text: '源码映射', link: '/projects/minimind/SOURCE_MAP' },
      ...minimindGroups.map(([section, text]) =>
        makeGroup('projects/minimind', '/projects/minimind', section, text)
      )
    ]
  }
]

export default defineConfig({
  lang: 'zh-CN',
  title: 'ML Algorithm Practice Lab',
  description: '面向手撕算法、深度学习训练机制和开源 LLM 项目拆解的学习站点',
  base,
  cleanUrls: false,
  ignoreDeadLinks: true,
  markdown: {
    math: true
  },
  themeConfig: {
    nav: [
      { text: '开始练习', link: '/ml_dl_handwriting/MODULE_INDEX' },
      { text: 'MiniMind 拆解', link: '/projects/minimind/MODULE_INDEX' },
      { text: 'GitHub', link: 'https://github.com/haibarazz/ml-algo-practice-lab' }
    ],
    sidebar,
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '没有找到相关结果',
            resetButtonTitle: '清除查询',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/haibarazz/ml-algo-practice-lab' }
    ],
    editLink: {
      pattern: 'https://github.com/haibarazz/ml-algo-practice-lab/blob/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026-present'
    }
  }
})
