import { renderContactSection, renderDocument } from "../components/layout.mjs";

export function renderAboutPage() {
  const body = `<main id="main">
    <section class="about-hero page-shell" aria-labelledby="about-page-title">
      <img class="about-avatar reveal" src="/assets/avatar.png" alt="Illustrated portrait of Yan Liang" width="84" height="84" fetchpriority="high">
      <p class="eyebrow reveal">A little more about me</p>
      <h1 id="about-page-title" class="reveal">Here is the<br><em>short version.</em></h1>
      <p class="about-hero-text reveal">I live in the Netherlands. I work on products and data. I write when I need to understand something better.</p>

      <div class="story-strip reveal" aria-label="Three notes about Yan">
        <article class="story-note story-note-product"><span>01</span><strong>Product work</strong><p>Start with the problem.</p></article>
        <article class="story-note story-note-writing"><span>02</span><strong>Writing</strong><p>Follow the thought.</p></article>
        <article class="story-note story-note-home"><span>03</span><strong>The Netherlands</strong><p>Home for now.</p></article>
      </div>
    </section>

    <div class="ticker-wrap" aria-hidden="true"><div class="ticker"><div>Curiosity <i></i> Clear thinking <i></i> Useful details <i></i> Quiet work <i></i> Curiosity <i></i> Clear thinking <i></i> Useful details <i></i> Quiet work <i></i> Curiosity <i></i> Clear thinking <i></i> Useful details <i></i> Quiet work <i></i></div></div></div>

    <section class="personal-section page-shell" aria-labelledby="personal-title">
      <header class="personal-intro reveal">
        <p class="eyebrow">Beyond the portfolio</p>
        <h2 id="personal-title">A few things<br>I care about</h2>
        <p>This is the part that rarely fits inside a project card.</p>
      </header>

      <div class="personal-grid">
        <article class="personal-card personal-card-read reveal"><p class="personal-label">On my desk</p><h3>The Psychology of Money</h3><div class="book-cover" aria-hidden="true"><span>The Psychology<br>of Money</span></div></article>
        <article class="personal-card personal-card-care reveal"><p class="personal-label">At work</p><h3>Useful products</h3><p>I care about privacy. I like clear choices. I want people to understand what a product is doing.</p></article>
        <article class="personal-card personal-card-tools reveal"><p class="personal-label">Tools I reach for</p><ul class="tool-list" aria-label="Tools Yan uses"><li>Product design</li><li>Python</li><li>Graph databases</li><li>Plain language</li></ul></article>
        <article class="personal-card personal-card-offline reveal"><p class="personal-label">Away from the screen</p><h3>Walks and stories</h3><p>I read fiction. I write short pieces. A long walk usually helps when an idea gets stuck.</p><div class="quiet-tags" aria-label="Personal interests"><span>Night owl</span><span>Traveller</span><span>Writer</span></div></article>
      </div>
    </section>

    <section class="principles-section page-shell" aria-labelledby="principles-title">
      <header class="principles-intro reveal"><p class="eyebrow">How I work</p><h2 id="principles-title">Simple rules I come back to</h2><p>Nothing fancy. These rules help me make progress when the answer is not clear yet.</p></header>
      <ol class="principle-list">
        <li class="reveal"><span>01</span><h3>Start with the problem</h3><p>I try to understand what is hard before I think about the interface.</p></li>
        <li class="reveal"><span>02</span><h3>Show how it works</h3><p>People should be able to follow the path. Important choices should not stay hidden.</p></li>
        <li class="reveal"><span>03</span><h3>Make something small</h3><p>A small working version teaches me more than a large plan.</p></li>
      </ol>
    </section>

    ${renderContactSection({ centered: true })}
  </main>`;

  return renderDocument({ title: "About", description: "A little more about Yan Liang and how he works.", path: "/about/", current: "about", body, bodyClass: "about-page" });
}
