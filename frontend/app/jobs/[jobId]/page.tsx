import { ReplayClient } from "@/components/ReplayClient";

export default async function JobReplayPage({ params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await params;
  return <ReplayClient jobId={jobId} />;
}
