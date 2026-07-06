import type { LucideIcon } from "lucide-react";

type StatCardProps = {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  tone?: "default" | "danger" | "success" | "info" | "warning";
};

export function StatCard({ label, value, icon: Icon, tone = "default" }: StatCardProps) {
  return (
    <article className={`stat-card ${tone === "default" ? "" : tone}`}>
      {Icon ? (
        <span className="stat-icon">
          <Icon size={19} />
        </span>
      ) : null}
      <div className="stat-body">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </article>
  );
}
