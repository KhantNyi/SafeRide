"use client";

import { useEffect, useRef, useState } from "react";
import { Maximize, Minimize } from "lucide-react";

export function VideoFullscreenButton() {
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [supported, setSupported] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setSupported(Boolean(document.fullscreenEnabled));
    const update = () => setFullscreen(document.fullscreenElement === buttonRef.current?.parentElement);
    document.addEventListener("fullscreenchange", update);
    return () => document.removeEventListener("fullscreenchange", update);
  }, []);

  async function toggleFullscreen() {
    const container = buttonRef.current?.parentElement;
    if (!container) return;
    setError("");
    try {
      if (document.fullscreenElement === container) await document.exitFullscreen();
      else await container.requestFullscreen();
    } catch {
      setError("Fullscreen could not open. Please try again.");
    }
  }

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        className="video-fullscreen-button"
        disabled={!supported}
        aria-label={fullscreen ? "Exit fullscreen" : "Fullscreen with detection boxes"}
        title={supported ? (fullscreen ? "Exit fullscreen" : "Fullscreen with detection boxes") : "Fullscreen overlays are unavailable in this browser"}
        onClick={toggleFullscreen}
      >
        {fullscreen ? <Minimize size={20} /> : <Maximize size={20} />}
      </button>
      {error ? <span className="video-fullscreen-error" role="alert">{error}</span> : null}
    </>
  );
}
