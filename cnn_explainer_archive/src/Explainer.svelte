<!-- this fiel again just imports the actual Overview -->
<script>
  import Overview from './overview/Overview.svelte';

  // this is not used
  import { cnnStore } from './stores.js';

  // Enum to control the displaying view
  const View = {
    OVERVIEW: 'overview',
    LAYERVIEW: 'layerview',
    DETAILVIEW: 'detailview'
  };

  let mainView = View.OVERVIEW;

  /* Example to read loaded cnn in other components:
  $: if ( $cnnStore.length != 0) {
    console.log($cnnStore);
  }
  */

</script>

<!-- 
  style in svelt are scoped to local component,
  so we can use the same class name in different components.
-->
<style>
#explainer {
  width: 100%;
  padding: 0;
  margin: auto;
  /* outline: 1px solid var(--g-dark-gray); */
}
</style>

<div id='explainer'>
    <Overview />
</div>