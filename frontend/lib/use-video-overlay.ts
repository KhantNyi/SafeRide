"use client";

import { useEffect, type RefObject } from "react";

export function useVideoOverlay(ref: RefObject<HTMLVideoElement | null>, draw: () => void, source?: string | null) {
  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    let callback = 0;
    let animation = 0;
    let stopped = false;
    const cancel = () => {
      if (callback) video.cancelVideoFrameCallback(callback);
      if (animation) cancelAnimationFrame(animation);
      callback = animation = 0;
    };
    const tick = () => {
      if (stopped) return;
      draw();
      if (video.paused || video.ended) return;
      if (typeof video.requestVideoFrameCallback === "function") callback = video.requestVideoFrameCallback(tick);
      else animation = requestAnimationFrame(tick);
    };
    const start = () => { cancel(); tick(); };
    const stop = () => { cancel(); draw(); };
    video.addEventListener("play", start);
    video.addEventListener("pause", stop);
    video.addEventListener("ended", stop);
    const resize = new ResizeObserver(() => draw());
    if (video.parentElement) resize.observe(video.parentElement);
    document.addEventListener("fullscreenchange", start);
    start();
    return () => {
      stopped = true;
      cancel();
      video.removeEventListener("play", start);
      video.removeEventListener("pause", stop);
      video.removeEventListener("ended", stop);
      resize.disconnect();
      document.removeEventListener("fullscreenchange", start);
    };
  }, [ref, draw, source]);
}
