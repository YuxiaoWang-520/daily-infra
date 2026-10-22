# Codex Long-Run Progress Log

- Original goal: 完成起步阶段前 10 篇高质量 AI Infra Wiki，逐篇验证审查并同步到私密 GitHub 仓库。
- Initialized at (UTC): 2026-10-22T10:10:25Z
- Detected stack: unknown

## Session 2026-10-22T10:10:25Z
- Feature: HARNESS-INIT
- Objective: Bootstrap long-running harness artifacts.
- Changes:
  - Added .codex-longrun/init.sh
  - Added .codex-longrun/feature_list.json
  - Added .codex-longrun/session_state.json
- Validation:
  - bootstrap script completed successfully
- Outcome: pass
- Next:
  - Execute F-000 or F-001 first, then continue one feature per session.
- Blockers:
- Could not auto-detect stack. Customize .codex-longrun/init.sh manually.

## Session 2026-10-22T10:11:59.833866+00:00
- Feature: PLAN
- Scope: 01—10, existing 01 counts; batch exception explicitly authorized.
- Plan: freeze scope and validate baseline; research/write/test/review/push each article; verify all ten.
- Constraints: CPU only; preserve playground/; no global dependency changes; per-article push.
- Next: baseline, then W02.

## Session 2026-10-22T10:12:44.835708+00:00
- Feature: BASELINE / W01 acceptance
- Validation: bash .codex-longrun/init.sh, exit 0; existing 26 CPU tests pass.
- Outcome: plan, baseline and existing article 01 accepted; no changes to article 01.
- Next: W02 broadcasting.

## Session 2026-10-22T10:16:51.640999+00:00
- Feature: W01 teaching revision (user steering)
- Changes: explain basic terms, row/column steps, 1D stride, view(-1), copy choices before formulas/source; update writing standard and template.
- Validation: 26 CPU tests, new examples and 18 local links PASS; two self-review rounds documented.
- Outcome: ready for immediate per-article push.
- Next: W02; use the slower question-driven narrative throughout.

## Session 2026-10-22T10:25:28.589033+00:00
- Feature: W02
- Changes: broadcasting lesson, two exercises, inline diagrams, pinned PyTorch source, references and review.
- Validation: 12 CPU tests incl. inference AD regression, Ruff and format, diagnostic assertions, 35 links and 8 pinned anchors PASS; independent Python review and re-review clear.
- Outcome: accepted locally; immediate article push in progress.
- Next: verify remote, mark W02, then W03 floating-point formats.

## Session 2026-10-22T10:38:55.606513+00:00
- Feature: W03
- Changes: gradual floating-point lesson, 2 exercises, 2 inline diagrams, 7 pinned source files, Mixed Precision Training v3 notes and acquisition metadata.
- Validation: 19 CPU tests, diagnostic assertions, Ruff and format PASS; independent Python review/re-review; exact tolerance and stable reference metrics fixed; 39 local links and 4 source anchors valid.
- Outcome: accepted locally; immediate chapter push pending.
- Next: verify remote and mark W03, then W04 Roofline.
- Research cache: .cache/research10/source-manifest.json and sources/ hold pinned PyTorch e149fba..., Triton c1641a..., FlashAttention 767386d... files. Paper PDFs/texts under .cache/research10/papers/ include mixed_precision, online_softmax, flashattention, gemm_anatomy. Roofline initial download timed out, resume then pdftotext exited 0; update its paper manifest before use.
- Reviewer continuity: reuse /root/python_bootstrap_review_fallback for narrow per-article Python reviews; fixed specialist unavailable. No new optional subagents.

## Session 2026-10-22T10:44:52.963095+00:00
- Feature: W03 remote acceptance
- Validation: GitHub main returned 5146360; all three published chapters pass 12+19+26=57 CPU tests in fresh interpreters.
- Outcome: W01-W03 complete and pushed; goal remains active, W04-W10 and FINAL remain false.
- Next: W04, Roofline. Primary paper cache roofline.txt sections 3/4 has been read; download resumed successfully, paper manifest corrected with initial error retained. Triton fused-softmax tutorial lines 40-105 read; note its comment says MN bytes where its calculation actually counts elements, do not copy that unit mistake. Preserve user playground/tensor.py.

## Session 2026-10-22T10:53:43.340493+00:00
- Feature: W04 remote acceptance
- Validation: 12 CPU tests, Ruff, 29 local links, 4 source anchors and 2 source hashes PASS; independent Python review and technical/teaching self-review complete.
- Outcome: 65ed2363b8eae9a81678165ceda87126924bd183 pushed and verified on GitHub main; 4/10 complete.
- Next: W05 mapping and coalescing geometry; protect playground/.
