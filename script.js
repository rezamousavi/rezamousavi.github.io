document.documentElement.classList.add("js");

const body = document.body;
const navToggle = document.querySelector(".nav-toggle");
const siteNav = document.querySelector(".site-nav");
const navLinks = [...document.querySelectorAll('.site-nav a[href^="#"]')];
const sections = [...document.querySelectorAll("section[id]")];
const reveals = [...document.querySelectorAll(".reveal")];

if (navToggle && siteNav) {
  navToggle.addEventListener("click", () => {
    const isOpen = siteNav.classList.toggle("is-open");
    navToggle.setAttribute("aria-expanded", String(isOpen));
  });

  siteNav.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    if (target.closest("a")) {
      siteNav.classList.remove("is-open");
      navToggle.setAttribute("aria-expanded", "false");
    }
  });
}

const setActiveNav = () => {
  const scrollPosition = window.scrollY + window.innerHeight * 0.28;
  let currentId = "";

  for (const section of sections) {
    if (scrollPosition >= section.offsetTop) {
      currentId = section.id;
    }
  }

  for (const link of navLinks) {
    const isActive = link.getAttribute("href") === `#${currentId}`;
    link.classList.toggle("is-active", isActive);
  }
};

const updateHeaderState = () => {
  body.classList.toggle("is-scrolled", window.scrollY > 18);
};

if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      }
    },
    {
      threshold: 0.18,
      rootMargin: "0px 0px -10% 0px",
    }
  );

  for (const element of reveals) {
    revealObserver.observe(element);
  }
} else {
  for (const element of reveals) {
    element.classList.add("is-visible");
  }
}

window.addEventListener("scroll", () => {
  updateHeaderState();
  setActiveNav();
});

window.addEventListener("resize", setActiveNav);

updateHeaderState();
setActiveNav();
