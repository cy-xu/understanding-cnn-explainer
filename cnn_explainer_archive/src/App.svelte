<!-- this is the main svelte app entrance.
	https://svelte.dev/repl is the playground where we can download 
	a barebone svelte app to see the init structure 
-->
<script>
	// the header and the content are two components
	// saved as two separate svelte files
	import Explainer from './Explainer.svelte';
	import Header from './Header.svelte';
</script>

<style>
</style>

<div id="app-page">
	<!-- the imported components are used this way in svelte -->
	<Header />
	<Explainer />
</div>
