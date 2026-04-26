<script lang="ts">
  import Header from './components/Header.svelte';
  import NeuralNetworkCard from './components/NeuralNetworkCard.svelte';
  import DataExplorer from './components/DataExplorer.svelte';
  import DrawClassify from './components/DrawClassify.svelte';
  import LossChart from './components/LossChart.svelte';
  import ValChart from './components/ValChart.svelte';
  import StatusBar from './components/StatusBar.svelte';
  import { cfg, ui, chartData } from './state.svelte';
  import { api, openTrainSocket, type TrainEvent } from './api';

  let totalSteps = $state(0);

  function handleEvent(ev: TrainEvent) {
    switch (ev.type) {
      case 'ping':
        return;
      case 'reset':
        chartData.steps = [];
        chartData.losses = [];
        chartData.epochs = [];
        chartData.valLosses = [];
        chartData.valAccs = [];
        ui.cycles = 0;
        return;
      case 'log':
        ui.status = ev.msg;
        return;
      case 'start':
        ui.training = true;
        totalSteps = ev.total_steps;
        if (ev.max_steps === 1) {
          ui.status = `single batch · step ${ev.starting_step} · epoch ${ev.starting_epoch}`;
        } else if (ev.max_epochs === 1) {
          ui.status = `single epoch · starting at step ${ev.starting_step} · epoch ${ev.starting_epoch}`;
        } else {
          ui.status = `training · ${ev.starting_step}/${ev.total_steps} steps · ${ev.epochs} epochs`;
        }
        return;
      case 'step':
        chartData.steps = [...chartData.steps, ev.step];
        chartData.losses = [...chartData.losses, ev.train_loss];
        ui.cycles = ev.step;
        ui.status = `epoch ${ev.epoch} · step ${ev.step} · train_loss ${ev.train_loss.toFixed(4)}`;
        return;
      case 'epoch':
        chartData.epochs = [...chartData.epochs, ev.epoch];
        chartData.valLosses = [...chartData.valLosses, ev.val_loss];
        chartData.valAccs = [...chartData.valAccs, ev.val_acc];
        ui.status = `epoch ${ev.epoch} done · val_loss ${ev.val_loss.toFixed(4)} · val_acc ${ev.val_acc.toFixed(4)} · ${ev.seconds.toFixed(1)}s`;
        return;
      case 'checkpoint':
        ui.checkpointBadge = `${ev.name} (${ev.val_acc.toFixed(3)})`;
        refreshCheckpoints();
        return;
      case 'paused':
        ui.training = false;
        ui.status = `paused after single batch · step ${ev.step} · epoch ${ev.epoch}`;
        return;
      case 'done':
        ui.training = false;
        ui.isContinuous = false;
        ui.status = `done · best val_acc ${ev.best_acc.toFixed(4)}`;
        return;
      case 'stopped':
        ui.training = false;
        ui.isContinuous = false;
        ui.status = 'stopped';
        return;
      case 'error':
        ui.training = false;
        ui.isContinuous = false;
        ui.status = `ERROR: ${ev.msg}`;
        return;
    }
  }

  async function refreshCheckpoints() {
    const { files, current } = await api.checkpoints();
    ui.checkpoints = files;
    if (current) {
      ui.selectedCheckpoint = current;
      ui.checkpointBadge = current;
    } else if (!ui.selectedCheckpoint && files.length) {
      ui.selectedCheckpoint = files[0];
    }
  }

  $effect(() => {
    (async () => {
      try {
        const [{ device }, session] = await Promise.all([
          api.device(),
          api.session(),
          refreshCheckpoints(),
        ]);
        ui.device = device;
        if (session.has_session && session.model) {
          cfg.model = session.model as 'mlp' | 'cnn';
          ui.cycles = session.step;
          ui.status = `resumed · ${session.model} · ${session.step.toLocaleString()} cycles`;
        } else {
          ui.status = 'Network initialized with random weights. Ready to start training.';
        }
      } catch (e) {
        const msg = (e as Error).message || String(e);
        ui.status = `backend unreachable: ${msg} — is the server on :5041?`;
        console.error('initial fetch failed', e);
      }
    })();

    const ws = openTrainSocket(handleEvent);
    ws.onerror = () => {
      ui.status = 'websocket error — check the server log';
      console.error('websocket error');
    };
    ws.onclose = () => {
      if (ui.status !== 'ready' && !ui.status.startsWith('websocket'))
        ui.status = 'websocket closed';
    };
    return () => ws.close();
  });
</script>

<div class="h-screen w-screen flex flex-col overflow-hidden">
  <Header />
  <main class="flex-1 min-h-0 flex flex-col gap-2 p-2">
    <div class="flex-1 min-h-0 flex gap-2">
      <NeuralNetworkCard />
      <DataExplorer />
      <DrawClassify />
    </div>
    <div class="h-56 shrink-0 flex gap-2">
      <LossChart />
      <ValChart />
    </div>
  </main>
  <StatusBar />
</div>
