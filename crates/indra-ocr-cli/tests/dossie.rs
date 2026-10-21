//! The CLI on real PDFs that are not in the repo. A six-page text-layer
//! dossier (INDRA_TEST_DOSSIE): letter-spaced headings, light-grey footers,
//! wrapped table cells, kicker headings and lists with drawn bullets. A scanned
//! flyer (INDRA_TEST_OCR_PDF). Without the variable, or without pdfium, a test
//! skips.

use std::process::Command;

/// Markdown and stderr of the CLI on the PDF that ``var`` names; None skips.
fn run(var: &str) -> Option<(String, String)> {
    let Some(pdf) = std::env::var_os(var) else {
        eprintln!("skipped: set {var} to the PDF");
        return None;
    };
    let out = Command::new(env!("CARGO_BIN_EXE_indra-ocr")).arg(&pdf).arg("--timings").output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    if err.contains("cannot load pdfium") {
        eprintln!("skipped: {err}");
        return None;
    }
    assert!(out.status.success(), "{err}");
    Some((String::from_utf8(out.stdout).unwrap(), err))
}

/// A scanned two-page flyer (INDRA_TEST_OCR_PDF): lists whose drawn bullets,
/// a circled check and circled digits, OCR reads as characters.
#[test]
fn flyer_lists() {
    let Some((md, _)) = run("INDRA_TEST_OCR_PDF") else { return };
    let has = |s: &str| assert!(md.contains(s), "missing {s:?} in:\n{md}");
    has("## Com a Tag Porto Bank, você ganha tempo");
    has("\n- Uso ilimitado em 100% dos pedágios");
    has("\n- Pague somente o que usar");
    has("\n1. Em “Atalhos”");
    has("\n2. Toque em \"Ativar tag\"");
    has("\n4. Prontinho");
    assert!(!md.contains("- d ") && !md.contains("- @"), "icon characters left in:\n{md}");
}

#[test]
fn dossie_markdown() {
    let Some((md, err)) = run("INDRA_TEST_DOSSIE") else { return };
    let has = |s: &str| assert!(md.contains(s), "missing {s:?} in:\n{md}");
    let lacks = |s: &str| assert!(!md.contains(s), "unexpected {s:?} in:\n{md}");

    // every page from the text layer (the light-grey footer has ink)
    assert!(!err.contains(": ocr,"), "a page went to OCR:\n{err}");

    // letter-spaced headings: spurious spaces gone, word spaces kept
    has("## RESUMO EXECUTIVO");
    has("IBM GUARDIUM® EMBARCADO");
    has("## NOME DA SOLUÇÃO DA NOTORIUN");
    has("## EXEMPLO · MAPA (DEFESA AGROPECUÁRIA)");
    has("PARA AGREGAR VALOR");
    lacks("R E S U M O");

    // a lone "." line joins its word; kicker and title are headings of their own
    has("## FIDACTA.\n");
    has("## O CIBERFÍSICO BASEADO NOS ATOS ADMINISTRATIVOS E COMO O IBM GUARDIUM IMPEDE FRAUDES\n\n## O ato administrativo é um dado;");
    has("## COMO A NOTORIUN UTILIZARÁ AS SOLUÇÕES IBM PARA AGREGAR VALOR AO PORTFÓLIO\n\n## A IBM entrega a captura");

    // wrapped cells: one table row per row, the cell text whole
    has("<tr><th>COMPONENTE INTEGRADO DE RESULTADO</th><th>O QUE A IBM ENTREGA (EMBARCADO)</th>");
    has(
        "<td>Antifraude e detecção (2ª linha de defesa)</td><td>DAM em tempo real; detecção de anomalia/outlier por ML; alerta por política</td>",
    );
    has("<td>Conformidade e tratamento lícito do dado</td>");
    lacks("<tr><td>defesa)</td>");

    // drawn bullets: one item per line, and no check marks read as "4"
    has("\n- Vê todo ato no dado.");
    has("\n- Preserva o ato de forma resistente a adulteração.");
    has("\n- Opera na 2ª linha de defesa:");
    has("\n- Tratamento de dado conforme LGPD");
    lacks("4 Dispositivo");

    // reading order: lists before what follows them, footer last
    let at = |s: &str| md.find(s).unwrap_or_else(|| panic!("missing {s:?}"));
    assert!(at("- Descobre e classifica") < at("## O salto ciberfísico"));
    assert!(at("- Tratamento de dado conforme LGPD") < at("Notoriun · FIDACTA · IBM Guardium® embarcado"));
}
