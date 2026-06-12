"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { AlertTriangle, Camera, LayoutDashboard, Upload } from "lucide-react";

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/upload", label: "Analysis", icon: Upload },
  { href: "/violations", label: "Violations", icon: AlertTriangle }
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link href="/dashboard" className="brand" aria-label="SafeRide dashboard">
          <span className="brand-mark">
            <Camera size={22} strokeWidth={2.2} />
          </span>
          <span>
            <strong>SafeRide</strong>
            <small>Vision Console</small>
          </span>
        </Link>

        <nav className="nav-list" aria-label="Primary navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
            return (
              <Link href={item.href} className={active ? "active" : ""} aria-current={active ? "page" : undefined} key={item.href}>
                <Icon size={18} />
                {item.label}
              </Link>
            );
          })}
        </nav>
      </aside>

      <main className="main-panel">{children}</main>
    </div>
  );
}
