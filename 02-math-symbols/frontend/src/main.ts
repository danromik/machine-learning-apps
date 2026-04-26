import { mount } from 'svelte';
import App from './App.svelte';
import './app.css';
import { initTheme } from './theme';

// Set the data-theme attribute before any component mounts so the
// ThemeSelector's initial value matches the actually-applied theme.
initTheme();

const app = mount(App, { target: document.getElementById('app')! });
export default app;
