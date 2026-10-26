# 仓库接入与发布

## 固定目的地

- 本地仓库：`/Users/wangyuxiao/Documents/ChatGPT/daily-infra`。
- GitHub：`YuxiaoWang-520/daily-infra`，保持私密。
- 目标分支：本地 `main` → `origin/main`。
- 仓库级作者身份：`YuxiaoWang-520 <13917046839@163.com>`。

这些是用户指定的发布配置，不是来源项目资料。所有 Git 命令显式指定目标仓库的工作目录；在其他项目调用本技能时，不操作来源项目的分支、远端或暂存区。目的仓库不存在时报告阻塞，不自行创建或替换远端仓库。

开始前读取目标仓库的 `AGENTS.md`、`docs/EDITORIAL.md`、`docs/DAILY_TASK.md`、`wiki/index.md`、`knowledge/index.md` 和相关模板；实际内容优先于本文件中的历史假设。

## 放到哪里

| 情况 | 处理方式 |
| --- | --- |
| 同一原理已在 Wiki 或知识索引中讲清 | 无新增知识则不改；有缺口则补充已有文章，同时更新其资料、审查和必要验证。 |
| 工作中提炼出独立的通用技术问题 | 使用 `knowledge/YYYY/MM/YYYY-MM-DD-topic/`，更新 `knowledge/index.md`。 |
| 用户明确要求完整 Wiki 章节 | 使用 `templates/article.md`，遵守完整 Wiki 的证据、图解、习题和日更频率要求；不要擅自挤占或改动排程。 |
| 尚有证据或验证缺口 | 已脱敏内容放入 `drafts/knowledge/YYYY-MM-DD-topic/`，保留 `type: knowledge`，不进发布索引，不推送未经审查的草稿。 |

使用实际 `Asia/Shanghai` 日期；`topic` 用知识主题的英文短名，不用项目或客户标识。按需知识文档不占工作日日更名额，不改动每日任务或自动化配置；同主题重复调用恢复原有条目，避免重复归档。

按需草稿由本知识沉淀任务恢复，日更任务只恢复 Wiki 草稿；两类任务不互相接管。正式 Wiki 草稿仍按 `docs/EDITORIAL.md` 处理。

每篇按需知识文档至少有：

```text
README.md       # 通用技术问题、解释、例子、工程取舍与验证边界
references.md   # 论断、公开来源版本、自己的阅读笔记及证据对应
review.md       # 技术、教学及脱敏审查，实际验证与未验证范围
```

代码、图源和合成验证记录分别按需存入 `code/`、`assets/`、`evidence/`。不创建空目录、空练习或没有实际内容的证据文件。知识文档使用 `templates/knowledge.md`；参考资料可改编 `templates/references.md`，删除不适用字段。审查按实际内容写，不机械勾选完整 Wiki 才适用的“两道练习”等要求。

上游公开仓库缓存放 `.cache/upstream/`，发布目录保留相关证据与来源说明；内部原始材料不放入这个缓存。不要访问或暂存用户的 `playground/`。

## 提交与同步

用户要求知识沉淀或调用本技能，且未要求仅草稿时，已授权完成脱敏和审查后的提交推送。遵循 `docs/DAILY_TASK.md` 的身份、远端和同步核验；本按需流程的内容白名单为本篇已审查目录、对应索引及必要的已有文章修正，不局限于日更流程的 `wiki/` 路径。

“演练”“模拟”“dry run”“只展示产物”覆盖默认发布流程：只在回复展示拟稿和动作，不写仓库、不更新索引、不执行 Git；后续明确要求实际执行时才继续。“仅草稿”可以保存已脱敏草稿，但不提交推送。正常发布已获授权，无需再确认。

1. 检查分支、读取和推送 URL、GitHub 私密属性、有效 Git 身份、工作区、暂存区与未推送提交。URL 必须精确指向上述仓库，不能只靠目录名判断目的地。需要纠正 Git 身份时只设置仓库级配置，不改全局公司身份。
2. 获取远端状态；只在安全时快进同步。存在分歧、无关未推送提交、无关暂存或同文件冲突时保留内容并报告，不把别人的工作混入提交。
3. 实际完成两轮审查、脱敏检查、必要验证和链接检查。对要更新的旧文同样检查新增证据，不只检查新文。
4. 在完整差异中检查是否有项目标识、凭据或原始材料，明确本次文件清单后，用具体路径 `git add -- <paths>`。不使用 `git add .`／`git add -A`；缓存、草稿、IDE 配置和无关文件不入库。
5. 检查完整暂存差异并运行 `git diff --cached --check`。使用 conventional commit，标题只描述知识主题，例如 `docs: explain asynchronous timing boundaries`。
6. 正常推送到 `origin/main`，然后通过 GitHub 核验目标分支包含该提交，记录提交 SHA 和链接。已经提交或已推送的同一变更不重复创建提交。

不强推、不自动 stash、rebase、reset 或清理文件；不创建 PR、不部署、不改可见性、不向他人发消息。认证、网络或远端核验失败时保留已脱敏文档及已有提交，明确报告“已归档”或“已提交未推送”和失败步骤，不声称同步完成。

## 技能本身的维护

Git 管理的源文件在 `skills/infra-knowledge-capture/`。本机用户级技能入口为 `~/.agents/skills/infra-knowledge-capture`，通过软链接指向这份源文件；修改时更新仓库内版本，避免维护两份副本。目标入口已存在且不是该技能时不覆盖。

用户级路径和软链接发现方式在 2026-10-26 核对过 [OpenAI Docs：Build skills](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)。更换机器后按实际环境重新建立入口，不改其他技能或配置。
