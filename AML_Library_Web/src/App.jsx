import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import DocumentViewer from './pages/DocumentViewer'
import MasterRedFlags from './pages/MasterRedFlags'
import Search from './pages/Search'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/master" element={<MasterRedFlags />} />
          <Route path="/document/:docId" element={<DocumentViewer />} />
          <Route path="/search" element={<Search />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App
