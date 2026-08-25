document.documentElement.classList.add("js");

const year = document.querySelector("[data-year]");
if (year) year.textContent = new Date().getFullYear();

const siteHeader = document.querySelector(".site-header");
if (siteHeader) {
  const updateNavigationSurface = () => siteHeader.classList.toggle("is-scrolled", window.scrollY > 24);
  updateNavigationSurface();
  window.addEventListener("scroll", updateNavigationSurface, { passive: true });
}

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const revealItems = document.querySelectorAll(".reveal, .reveal-line");

if (reducedMotion || !("IntersectionObserver" in window)) {
  revealItems.forEach((item) => item.classList.add("is-visible"));
} else {
  const revealObserver = new IntersectionObserver((entries, observer) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add("is-visible");
      observer.unobserve(entry.target);
    }
  }, { rootMargin: "0px 0px -7%", threshold: 0.06 });

  revealItems.forEach((item) => revealObserver.observe(item));
}

const dotGridScene = document.querySelector("[data-dot-grid-scene]");
const dotGrid = dotGridScene?.querySelector("[data-dot-grid]");
const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

if (dotGridScene && dotGrid) {
  const gridContext = dotGrid.getContext("2d");
  const gridSpacing = 48;
  const interactionRadius = 180;
  const pointer = { x: 0, y: 0, strength: 0, targetStrength: 0 };
  let gridBounds;
  let gridFrame;

  const drawDotGrid = () => {
    gridContext.clearRect(0, 0, gridBounds.width, gridBounds.height);
    const horizontalOffset = (gridBounds.width % gridSpacing) / 2;
    const verticalOffset = (gridBounds.height % gridSpacing) / 2;

    for (let y = verticalOffset; y <= gridBounds.height; y += gridSpacing) {
      for (let x = horizontalOffset; x <= gridBounds.width; x += gridSpacing) {
        const distanceX = pointer.x - x;
        const distanceY = pointer.y - y;
        const distance = Math.hypot(distanceX, distanceY);
        const proximity = finePointer && !reducedMotion
          ? Math.max(0, 1 - distance / interactionRadius) * pointer.strength
          : 0;
        const easedProximity = proximity * proximity;
        const safeDistance = distance || 1;
        const pull = easedProximity * 8;
        const dotX = x + distanceX / safeDistance * pull;
        const dotY = y + distanceY / safeDistance * pull;

        gridContext.beginPath();
        gridContext.arc(dotX, dotY, 1 + easedProximity * 1.6, 0, Math.PI * 2);
        gridContext.fillStyle = `rgba(185, 236, 145, ${.12 + easedProximity * .58})`;
        gridContext.fill();
      }
    }
  };

  const animateDotGrid = () => {
    pointer.strength += (pointer.targetStrength - pointer.strength) * .16;
    drawDotGrid();
    if (Math.abs(pointer.targetStrength - pointer.strength) > .01) {
      gridFrame = window.requestAnimationFrame(animateDotGrid);
    } else {
      pointer.strength = pointer.targetStrength;
      gridFrame = undefined;
    }
  };

  const queueGridFrame = () => {
    if (!gridFrame) gridFrame = window.requestAnimationFrame(animateDotGrid);
  };

  const sizeDotGrid = () => {
    gridBounds = dotGrid.getBoundingClientRect();
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    dotGrid.width = Math.round(gridBounds.width * pixelRatio);
    dotGrid.height = Math.round(gridBounds.height * pixelRatio);
    gridContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    drawDotGrid();
  };

  sizeDotGrid();
  new ResizeObserver(sizeDotGrid).observe(dotGridScene);

  if (finePointer && !reducedMotion) {
    dotGridScene.addEventListener("pointerenter", (event) => {
      pointer.x = event.clientX - gridBounds.left;
      pointer.y = event.clientY - gridBounds.top;
      pointer.targetStrength = 1;
      queueGridFrame();
    });
    dotGridScene.addEventListener("pointermove", (event) => {
      pointer.x = event.clientX - gridBounds.left;
      pointer.y = event.clientY - gridBounds.top;
      queueGridFrame();
    });
    dotGridScene.addEventListener("pointerleave", () => {
      pointer.targetStrength = 0;
      queueGridFrame();
    });
  }
}

const writingLists = document.querySelectorAll(".writing-list-layout");

for (const layout of writingLists) {
  const preview = layout.querySelector(".writing-preview");
  const topic = preview?.querySelector("[data-preview-topic]");
  const copy = preview?.querySelector("[data-preview-copy]");
  const rows = layout.querySelectorAll("[data-writing-preview]");
  if (!preview || !topic || !copy) continue;
  const narrowLayout = window.matchMedia("(max-width: 900px)");
  let activeRow = rows[0];
  let previewTimer;

  // On small screens the preview follows its selected title so text and imagery stay connected.
  const placePreview = () => {
    if (narrowLayout.matches && activeRow) activeRow.insertAdjacentElement("afterend", preview);
    else layout.append(preview);
  };

  const showPreview = (row) => {
    activeRow = row;
    rows.forEach((item) => item.classList.toggle("is-active", item === row));
    placePreview();
    preview.classList.add("is-changing");
    window.clearTimeout(previewTimer);
    previewTimer = window.setTimeout(() => {
      topic.textContent = row.dataset.writingTopic;
      copy.textContent = row.dataset.writingPreview;
      preview.classList.remove("is-changing");
    }, reducedMotion ? 0 : 140);
  };

  rows.forEach((row) => {
    row.addEventListener("pointerenter", () => showPreview(row));
    row.addEventListener("focus", () => showPreview(row));
  });

  placePreview();
  narrowLayout.addEventListener("change", placePreview);
}
