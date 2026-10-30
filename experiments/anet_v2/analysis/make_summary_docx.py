"""Generate the A-Net experiment summary DOCX without external dependencies."""

from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "A-Net_Experiment_Summary.docx"


def r(text: str, bold: bool = False, color: str | None = None, size: int | None = None) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if color:
        props.append(f'<w:color w:val="{color}"/>')
    if size:
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


class Doc:
    def __init__(self) -> None:
        self.body: list[str] = []

    def p(self, text: str = "", style: str | None = None, bold: bool = False,
          color: str | None = None, size: int | None = None, align: str | None = None) -> None:
        ppr = []
        if style:
            ppr.append(f'<w:pStyle w:val="{style}"/>')
        if align:
            ppr.append(f'<w:jc w:val="{align}"/>')
        ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
        self.body.append(f"<w:p>{ppr_xml}{r(text, bold, color, size)}</w:p>")

    def bullet(self, text: str) -> None:
        ppr = '<w:pPr><w:ind w:left="420" w:hanging="220"/></w:pPr>'
        self.body.append(f"<w:p>{ppr}{r('• ' + text)}</w:p>")

    def page_break(self) -> None:
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def table(self, headers: list[str], rows: list[list[str]], widths: list[int] | None = None,
              font_size: int = 16) -> None:
        n = len(headers)
        if widths is None:
            widths = [9000 // n] * n
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        xml = [
            '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
            '<w:tblW w:w="0" w:type="auto"/><w:tblLayout w:type="fixed"/>'
            '<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="70" w:type="dxa"/>'
            '<w:bottom w:w="40" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tblCellMar>'
            '</w:tblPr>',
            f"<w:tblGrid>{grid}</w:tblGrid>",
        ]
        def cell(text: str, width: int, header: bool = False) -> str:
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/>' if header else ""
            tcpr = f'<w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade}</w:tcPr>'
            body = r(text, bold=header, size=font_size)
            return f"<w:tc>{tcpr}<w:p><w:pPr><w:pStyle w:val=\"TableText\"/></w:pPr>{body}</w:p></w:tc>"
        xml.append("<w:tr>")
        for h, w in zip(headers, widths):
            xml.append(cell(h, w, True))
        xml.append("</w:tr>")
        for row in rows:
            xml.append("<w:tr>")
            for text, w in zip(row, widths):
                xml.append(cell(str(text), w, False))
            xml.append("</w:tr>")
        xml.append("</w:tbl>")
        self.body.append("".join(xml))

    def save(self) -> None:
        sect = (
            '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
            'w:header="567" w:footer="567" w:gutter="0"/>'
            '<w:cols w:space="708"/><w:docGrid w:linePitch="312"/></w:sectPr>'
        )
        document = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{''.join(self.body)}{sect}</w:body></w:document>"
        )
        styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults><w:rPrDefault><w:rPr>
    <w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/>
    <w:sz w:val="21"/><w:szCs w:val="21"/>
  </w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="80" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
    <w:pPr><w:spacing w:before="1800" w:after="240"/><w:jc w:val="center"/></w:pPr>
    <w:rPr><w:b/><w:color w:val="1F3864"/><w:sz w:val="44"/><w:szCs w:val="44"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:after="120"/><w:jc w:val="center"/></w:pPr>
    <w:rPr><w:color w:val="595959"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
    <w:pPr><w:keepNext/><w:spacing w:before="360" w:after="140"/><w:outlineLvl w:val="0"/></w:pPr>
    <w:rPr><w:b/><w:color w:val="1F4E79"/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
    <w:pPr><w:keepNext/><w:spacing w:before="220" w:after="100"/><w:outlineLvl w:val="1"/></w:pPr>
    <w:rPr><w:b/><w:color w:val="2E74B5"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/>
    <w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>
    <w:pPr><w:spacing w:after="0" w:line="220" w:lineRule="auto"/></w:pPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/>
    <w:tblPr><w:tblBorders>
      <w:top w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>
      <w:left w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>
      <w:bottom w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>
      <w:right w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>
      <w:insideH w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>
      <w:insideV w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>
    </w:tblBorders></w:tblPr></w:style>
</w:styles>'''
        content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
        rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
        doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
        core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>A-Net 实验总结：结果、优势与代价</dc:title>
  <dc:creator>A-Net Experiment</dc:creator>
  <cp:lastModifiedBy>A-Net Experiment</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{date.today().isoformat()}T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{date.today().isoformat()}T00:00:00Z</dcterms:modified>
</cp:coreProperties>'''
        app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>DSH</Application>
</Properties>'''
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", content_types)
            z.writestr("_rels/.rels", rels)
            z.writestr("word/document.xml", document)
            z.writestr("word/styles.xml", styles)
            z.writestr("word/_rels/document.xml.rels", doc_rels)
            z.writestr("docProps/core.xml", core)
            z.writestr("docProps/app.xml", app)
        print(OUT, OUT.stat().st_size)


def main() -> None:
    d = Doc()
    d.p("A-Net 实验总结", "Title")
    d.p("结果、优势与代价", "Subtitle")
    d.p("截至 2026-10-30 · 分支 experiment/anet-v2 · 记录时提交 45ec348", "Subtitle")
    d.p("实验环境：RTX 5060 Laptop（8 GB 专用显存）· 15.18 GB RAM · PyTorch 2.14.1+cu130", "Subtitle")
    d.page_break()

    d.p("执行摘要", "Heading1")
    d.p("本报告汇总 A-Net v2 在 2D 合成任务、缓存基准、50M char-level 语言 pilot，以及激活参数/共享深度对照中的全部主要结果。结论分为“机制成立”“能力证据”“成本权衡”“尚未验证”四类，避免把 pilot 结果写成大模型结论。")
    d.bullet("参数所有权、引用、版本、缓存失效和隔离机制均已实现并通过测试；最新测试 75 passed。")
    d.bullet("参数共享能显著降低唯一参数：Stage 1 A-Net Full 0.651×；Stage 2 Share 0.656×；Stage 4 W=3 0.67×、W=4 0.63×。")
    d.bullet("在 50M pilot 中，Sparse A-Net 因有效深度仅 6 层而明显更差（2.141 bpc）；修正为 12 层 Shared-Depth A-Net 后，以 1.933 bpc 略优于 Transformer-12L 的 1.991 bpc（p≈0.046）。")
    d.bullet("该 50M 结果只是 pilot：3 seeds、3M tokens、char-level 三域数据；不是 MMLU/GSM8K/HumanEval 能力结论。")
    d.bullet("推理延迟只有条件性优势；显存和训练时间没有普遍优势。Sparse A-Net 训练峰值更低（1.69 vs 1.98 GB），但以深度/能力为代价。")
    d.bullet("2B 全参数训练在本机不可行（约 32 GB 训练状态 vs 8 GB GPU / 15 GB RAM）；1B 最终验证需要外部 8×H100 级资源。")

    d.p("1. 实验口径与边界", "Heading1")
    d.p("所有架构使用相同的唯一参数、激活参数、引用参数和存储统计口径。共享参数只在唯一参数量中计一次；参数引用总量允许重复；单次前向激活唯一参数按实际调用路径统计；存储按 FP32 唯一参数估算。")
    d.table(
        ["口径", "定义"],
        [
            ["唯一可训练参数", "去重后的 requires_grad 参数总量；共享参数只计一次"],
            ["参数引用总量", "按调用点累计，共享块被多个 workflow 调用会重复计入"],
            ["单次前向激活唯一参数", "一次 forward 中实际执行的唯一参数"],
            ["参数共享比例", "共享参数元素 / 唯一参数，或按具体对照说明"],
            ["实际存储", "唯一参数 × 4 bytes（FP32）"],
        ],
        [2600, 6400],
    )
    d.p("数据来源：Stage 1–4 为 2D 合成任务；50M pilot 为 byte-level Tiny Shakespeare + CPython 源码 + 合成数学；2B 仅做可行性审计，没有训练。")
    d.p("证据强度：Stage 1–4 使用 3–5 seeds；50M pilot 使用 3 seeds；p 值仅作 pilot 级参考。没有正式的多重比较校正或大模型复现。")

    d.p("2. 实验矩阵", "Heading1")
    d.table(
        ["实验", "规模", "状态", "主要用途"],
        [
            ["Stage 1", "360 runs", "完成", "宽度 scaling；精度/参数/延迟/吞吐"],
            ["Stage 2 耦合", "315 runs", "完成", "共享的负迁移/梯度冲突"],
            ["Stage 2 隔离", "315 runs", "完成", "detach_both 隔离对照"],
            ["Stage 3", "288 cache streams", "完成", "缓存正确性、命中、延迟、内存"],
            ["Stage 4", "540 runs", "完成", "share × cache × 宽度 × workflow 数"],
            ["50M 语言 pilot", "9 runs", "完成", "Transformer/Shared/Sparse A-Net"],
            ["激活匹配对照", "3 runs", "完成", "Transformer-12L vs Sparse A-Net"],
            ["对照 C 共享深度", "3 runs", "完成", "12 层 Shared-Depth A-Net vs Transformer-12L"],
            ["2B", "未执行", "不可行", "本机资源不足；只做可行性审计"],
        ],
        [2300, 2100, 1600, 3000],
    )

    d.p("3. Stage 1：基础架构 scaling", "Heading1")
    d.p("360 个 run，0 失败；3 seeds；h64/h256/h1024/h2048；3 个合成 2D 任务。下表为相对 Independent 的平均值。")
    d.table(
        ["模型", "平均精度", "唯一参数比", "激活参数比", "batch1 P50", "大 batch 吞吐"],
        [
            ["A-Net Basic", "-0.067 pp", "0.967×", "0.996×", "2.65×", "0.61×"],
            ["A-Net Full", "-0.216 pp", "0.651×", "0.996×", "4.75×", "0.47×"],
            ["Shared Backbone", "-1.367 pp", "0.341×", "1.000×", "相近", "相近"],
        ],
        [2200, 1600, 1500, 1500, 1400, 1400],
    )
    d.p("解释：A-Net 精度接近 Independent，但不是计算/延迟胜利；Shared Backbone 存储最少但精度损失最大。A-Net 位于“比 Independent 省存储、比 Shared 保精度”的中间位置。")
    d.p("重构后 A-Net Full 的 batch1 P50 从约 555 µs 降到约 198 µs（纯路径），但仍约为 Independent 的 2.3–2.6 倍。")

    d.p("4. Stage 2：共享的收益与隔离", "Heading1")
    d.p("耦合共享（315 runs）中，A-Net Share 相对 Independent 唯一参数 0.656×、激活参数 1.48×，但平均精度 -1.20 pp；Basic -0.95 pp、Adapter -1.63 pp。梯度 cosine 约 +0.025、冲突比例约 0.43；task0 负迁移 -2.14 pp，回归 MSE +0.0106。")
    d.p("隔离版（detach_both，315 runs）显著改善负迁移：")
    d.table(
        ["模式", "相对耦合版提升", "说明"],
        [
            ["pair Basic", "+0.856 pp", "隔离辅助 workflow"],
            ["pair Share", "+1.126 pp", "共享块冻结，避免冲突"],
            ["pair Adapter", "+1.493 pp", "adapter 也不接收冲突梯度"],
            ["dense Basic", "+1.599 pp", "更明显的隔离收益"],
            ["dense Share", "+1.681 pp", "dense 路由"],
            ["dense Adapter", "+1.750 pp", "dense 路由"],
        ],
        [2400, 2600, 4000],
    )
    d.p("隔离版绝对精度：Independent 95.87%、Basic 95.77%、Share 95.80%、Adapter 95.73%；pair Share 相对 Independent 仅 -0.07 pp。task0 低 0.22–0.29 pp，回归 MSE +0.0020–0.0022。")
    d.p("代价：detach_both 同时冻结共享块输出和辅助 workflow，共享块不再接收梯度；如果要求共享块继续学习，应改用 detach_aux 或梯度手术。")

    d.p("5. Stage 3：结果缓存", "Heading1")
    d.p("288 个 cache stream，0 失败。cache on/off 输出最大绝对差 0.0；参数更新后 100% 失效；H2D = 0；所有输入在 cuda:0。")
    d.table(
        ["指标/场景", "结果"],
        [
            ["命中率（Reuse）", "精确等于输入重复率：0/25/50/75%"],
            ["命中率（Full）", "额外 20%：pair 内共享块复用"],
            ["执行次数（75% 重复）", "Reuse 25%，Full 20%"],
            ["batch=1 延迟", "cache 更慢 1.14–1.85×"],
            ["2048 宽 / batch256 / 75%", "Reuse 0.68×，Full 0.67×"],
            ["2048 宽 / batch256 / 50%", "Full 0.98×，接近盈亏平衡"],
            ["缓存内存", "最高 256 MB；最多 786 次 eviction"],
            ["查找成本", "约 1.33 µs/次"],
        ],
        [3600, 5400],
    )
    d.p("结论：缓存不是普遍加速；只有大宽度、大 batch、高重复率才有净收益。默认 inference cache 关闭。")

    d.p("6. Stage 4：共享 × 复用 × 宽度 × workflow 数", "Heading1")
    d.p("540 个训练 run，0 失败；3 seeds；2×2 因子（share × cache）；W=2/3/4；宽度 128/512/2048。")
    d.table(
        ["配置", "唯一参数比（相对 Basic）", "精度效应"],
        [
            ["A-Net Share，W=2", "0.76×", "W=2 容量不足，+0.3–1.1 pp"],
            ["A-Net Share，W=3", "0.67×", "约中性（-0.16–+0.15 pp）"],
            ["A-Net Share，W=4", "0.63×", "约中性（-0.10–+0.13 pp）"],
        ],
        [3000, 2800, 3200],
    )
    d.p("Reuse 的精度效应严格为 0：训练每个 optimizer step 清空缓存，缓存只影响推理。交互项为 0：共享与复用没有额外精度协同。")
    d.p("Workflow 数：W=2 明显容量不足（平均 83.92% vs W=3 95.77%、W=4 95.87%）；W=4 相对 W=3 没有稳定精度收益，但唯一参数继续增加。")
    d.p("缓存对照（width=2048、W=3、medium）：batch=1 全部更慢；batch=256、75% 重复时 Reuse 0.68×、Full 0.67×；50% 重复时 Full 0.98×。")

    d.p("7. 50M 语言 pilot", "Heading1")
    d.p("这是本机可执行的最小语言代理实验，不是 MMLU/GSM8K/HumanEval 能力结论。数据为 Tiny Shakespeare、CPython 源码、合成数学；byte-level vocab=256；3 seeds；每 seed 1500 步、batch 16、seq 128，约 3M tokens；AdamW、FP32。")
    d.table(
        ["模型", "唯一参数", "单次激活参数", "有效深度", "平均 bpc"],
        [
            ["A-Net Shared-Depth", "50.18M", "37.50M", "12", "1.933 ± 0.067"],
            ["Transformer-12L", "37.50M", "37.50M", "12", "1.991 ± 0.033"],
            ["Shared Backbone", "50.18M", "50.18M", "16", "1.975 ± 0.028"],
            ["Transformer-16L", "49.92M", "49.92M", "16", "2.007 ± 0.055"],
            ["Sparse A-Net", "50.18M", "37.50M", "6", "2.141 ± 0.107"],
        ],
        [2800, 1500, 1600, 1300, 1800],
    )
    d.p("三架构原始对照：Transformer 2.007、Shared Backbone 1.975、Sparse A-Net 2.141。Sparse A-Net 更差的主要原因是有效深度只有 6 层，而不是“共享一定伤害能力”。")
    d.p("激活匹配对照：Transformer-12L（37.503M 激活）1.991 vs Sparse A-Net（37.502M 激活）2.141；差 +0.150 bpc，Welch p≈0.019。该对照排除了“只是激活参数少”，但仍存在深度/路由混淆。")

    d.p("8. 对照 C：Shared-Depth A-Net", "Heading1")
    d.p("为消除深度混淆，构造 8 个共享 block + 2 个 workflow 各 4 个专属 block 的 12 层路径：")
    d.p("embedding → 8 shared blocks → 4 task-workflow blocks → task head", "Normal")
    d.table(
        ["项目", "A-Net Shared-Depth", "Transformer-12L"],
        [
            ["唯一参数", "50,177,024", "37,503,168"],
            ["单次激活参数", "37,501,632", "37,503,168"],
            ["激活参数差", "1,536（0.0041%）", "—"],
            ["顺序深度", "12", "12"],
            ["共享比例", "49.5%", "0%"],
            ["参数引用总量", "75,002,496", "37,503,168"],
            ["FP32 唯一存储", "200.7 MB", "150.0 MB"],
        ],
        [2600, 3200, 3200],
    )
    d.table(
        ["模型", "text", "code", "math", "平均"],
        [
            ["A-Net Shared-Depth", "2.537", "1.736", "1.527", "1.933 ± 0.067"],
            ["Transformer-12L", "2.635", "1.815", "1.523", "1.991 ± 0.033"],
            ["差值（A-Net − 对照）", "-0.098", "-0.079", "+0.004", "-0.058"],
        ],
        [3000, 1300, 1300, 1300, 2100],
    )
    d.p("Welch t≈-3.35，p≈0.046；A-Net 在 text/code 更好，math 基本打平。该结果支持“共享深度可以在不牺牲深度/激活计算的情况下保持能力”，但只有 3 seeds、3M tokens，属于 pilot 级证据。")

    d.p("9. 成本：推理、显存、训练时间", "Heading1")
    d.table(
        ["指标", "A-Net Shared-Depth", "Transformer-12L", "结论"],
        [
            ["batch1 P50", "3.00 ms", "4.70 ms", "A-Net 0.64×，更快"],
            ["batch16 P50", "14.89 ms", "15.23 ms", "基本持平"],
            ["训练峰值显存", "1.59 GB", "1.51 GB", "A-Net 更高"],
            ["唯一参数存储", "200.7 MB", "150.0 MB", "A-Net 更高"],
            ["训练时间/seed", "100.3 s", "101.7 s", "约 1%，无实质差异"],
        ],
        [2600, 2200, 2200, 2000],
    )
    d.table(
        ["模型", "唯一参数", "单次激活", "训练峰值", "训练时间"],
        [
            ["Sparse A-Net", "50.18M", "37.50M", "1.69 GB", "94.6 s"],
            ["Transformer-16L", "49.92M", "49.92M", "1.98 GB", "121.7 s"],
            ["Shared Backbone", "50.18M", "50.18M", "2.11 GB", "120.8 s"],
            ["Shared-Depth A-Net", "50.18M", "37.50M", "1.59 GB", "100.3 s"],
            ["Transformer-12L", "37.50M", "37.50M", "1.51 GB", "101.7 s"],
        ],
        [2800, 1500, 1500, 1600, 1600],
    )
    d.p("")
    d.p("Sparse A-Net 的训练峰值和训练时间确实更低（1.69 GB / 94.6 s），但这是以只激活 2/3 workflow、有效深度 6 层、能力更差为代价。Shared-Depth A-Net 保住了深度/能力，但唯一参数更多，训练显存反而略高。Stage 1 中 A-Net 推理更慢、训练更慢；缓存只在特定场景回本。")

    d.p("10. 优势", "Heading1")
    for text in [
        "参数所有权、唯一存储、版本、授权调用和缓存失效机制正确，可审计；不是隐式全局共享。",
        "参数共享降低唯一参数：Stage 1 Full 0.651×；Stage 2 Share 0.656×；Stage 4 W=3 0.67×、W=4 0.63×。",
        "在 50M 同深度/同激活参数对照中，Shared-Depth A-Net 1.933 bpc 略优于 Transformer-12L 1.991 bpc（p≈0.046，pilot）。",
        "隔离（detach_both）显著改善多任务负迁移：pair Share 相对耦合 +1.13 pp，dense +1.68 pp；隔离版接近 Independent。",
        "结果缓存正确；高重复、大 batch、大宽度下执行次数和延迟下降（最高约 0.67×）。",
        "Sparse A-Net 训练显存/时间更低（1.69 GB/94.6 s），适合激活路径受限的场景，但能力下降。",
        "工程上 GPU-resident、H2D=0、显存守卫、五口径核算和 75 个测试，便于复现和审计。",
    ]:
        d.bullet(text)

    d.p("11. 劣势与代价", "Heading1")
    for text in [
        "A-Net 不是普遍的计算/延迟胜利：Stage 1 Basic/Full batch1 为 2.65×/4.75×，吞吐 0.61×/0.47×，训练时间 1.21×/1.33×。",
        "Sparse A-Net 把有效路径从 12 层降到 6 层，唯一参数 50M 但单次只激活 37.5M；能力明显更差（2.141 bpc）。",
        "Shared-Depth A-Net 的唯一参数反而比 Transformer-12L 多 12.7M，训练显存 1.59 vs 1.51 GB 更高；共享不是“更省显存”的同义词。",
        "训练时间没有普遍优势；Shared-Depth 100.3 vs 101.7 s，约 1% 差异。",
        "缓存有明显开销：batch=1 慢 1.14–1.85×，最高 256 MB，查找 1.33 µs/次；只有高重复/大 batch 回本。",
        "detach_both 冻结共享块，牺牲共享表示学习；要共享块继续学习必须换 detach_aux/梯度手术。",
        "Workflow 数存在容量阈值：W=2 对 3 任务容量不足；W=4 无稳定精度收益但参数更多。",
        "工程复杂度高：registry、版本、DAG、cache、隔离、设备守卫；比普通 Transformer 更难维护。",
        "50M pilot 只有 3 seeds、3M tokens、char-level 数据；p≈0.046 接近 0.05，不能当作最终显著性。",
        "没有 MMLU/GSM8K/HumanEval、SFT、1B/2B 实验；2B 本机不可行。",
    ]:
        d.bullet(text)

    d.p("12. 统计与证据强度", "Heading1")
    d.table(
        ["结果", "样本", "统计", "证据强度"],
        [
            ["Shared-Depth 优于 Transformer-12L", "3 seeds", "均值差 -0.058 bpc；p≈0.046", "pilot 级；接近阈值"],
            ["Sparse 差于 Transformer-12L", "3 seeds", "均值差 +0.150 bpc；p≈0.019", "pilot 级；有深度混淆"],
            ["Stage 1/2/4", "3–5 seeds", "均值/标准差；无正式校正", "趋势级"],
            ["Stage 3 cache", "288 streams", "输出等价、命中/延迟", "机制级，较强"],
            ["2B", "0", "未执行", "无能力证据"],
        ],
        [3200, 1600, 3000, 2200],
    )

    d.p("13. 结论", "Heading1")
    d.p("支持的结论：")
    d.bullet("A-Net v2 的参数共享、所有权、版本、缓存、隔离机制可以正确实现。")
    d.bullet("共享能降低唯一参数；在 W=3/4 下 Stage 4 唯一参数比约 0.67×/0.63×。")
    d.bullet("在 50M pilot 的同深度/同激活参数对照中，Shared-Depth A-Net 略优于 Transformer-12L（p≈0.046）。")
    d.bullet("隔离改善负迁移；缓存正确且在高重复/大 batch 下有条件收益。")
    d.p("不支持的结论：")
    d.bullet("A-Net 无条件优于 Transformer。")
    d.bullet("A-Net 计算更少、延迟更低或显存更省。")
    d.bullet("1B/2B 语言能力已超过 Transformer。")
    d.bullet("结果复用显著提升训练能力。")
    d.bullet("Sparse A-Net 的失败证明共享机制无效。")
    d.p("总体判断：A-Net 当前最可靠的定位是“以参数共享换唯一存储、以隔离换稳定、以缓存换特定场景执行”的架构；能力优势仅在 50M pilot 的 Shared-Depth 对照中初步出现，仍需更大规模复现。")

    d.p("14. 下一步", "Heading1")
    for text in [
        "补 5–10 seeds 和更长 token 预算，复现 Shared-Depth 的 0.058 bpc 差异。",
        "做 dense A-Net、shared-depth、不同共享位置/深度的消融，确认增益来自共享而非实现细节。",
        "在 1B 规模、20B–100B tokens、标准评测上验证；需要 8×H100 级资源。",
        "补 cache 的端到端推理基准和显存/吞吐测量；当前只证明条件性收益。",
        "把“训练显存低”限定为 Sparse A-Net 的代价性优势，不推广到 Shared-Depth。",
    ]:
        d.bullet(text)

    d.p("附录：产物与提交", "Heading1")
    d.table(
        ["文件/产物", "说明"],
        [
            ["README.md", "项目总览与最新结论"],
            ["experiments/anet_v2/STAGE1_REPORT.md", "Stage 1 报告"],
            ["experiments/anet_v2/STAGE2_REPORT.md", "Stage 2 耦合报告"],
            ["experiments/anet_v2/STAGE2_ISOLATED_REPORT.md", "Stage 2 隔离报告"],
            ["experiments/anet_v2/STAGE3_REPORT.md", "缓存报告"],
            ["experiments/anet_v2/STAGE4_REPORT.md", "Stage 4 因子报告"],
            ["experiments/anet_v2/lang50m/REPORT.md", "50M 三架构报告"],
            ["experiments/anet_v2/lang50m/ACTIVE_MATCH_REPORT.md", "激活匹配对照"],
            ["experiments/anet_v2/lang50m/SHARED_DEPTH_REPORT.md", "对照 C 共享深度"],
            ["experiments/anet_v2/scale2b/FEASIBILITY.md", "2B 可行性"],
        ],
        [5000, 4000],
    )
    d.p("分支：experiment/anet-v2；记录时提交：45ec348；测试：75 passed。所有原始结果和报告已推送到 GitHub 分支。")
    d.p("说明：本报告由实验数据自动汇总生成；没有使用小模型冒充 1B/2B，也没有把随机权重或 pilot 结果写成语言能力结论。")

    d.save()


if __name__ == "__main__":
    main()
